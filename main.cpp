#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <queue>
#include <memory>
#include <chrono>
#include <random>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <format>
#include <ranges>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cassert>

namespace ota_simulator {

// Forward declarations
class Device;
class OTAManager;
enum class UpdateState;

/**
 * Semantic versioning with rollback capability support
 * @intuition: Need structured version comparison for update eligibility and rollback decisions
 * @approach: Three-component versioning (major.minor.patch) with comparison operators
 * @complexity: Time O(1), Space O(1)
 */
struct FirmwareVersion {
    uint32_t major{0};
    uint32_t minor{0};
    uint32_t patch{0};
    
    constexpr auto operator<=>(const FirmwareVersion& other) const = default;
    constexpr bool operator==(const FirmwareVersion& other) const = default;
    
    [[nodiscard]] std::string to_string() const {
        return std::format("{}.{}.{}", major, minor, patch);
    }
    
    [[nodiscard]] static FirmwareVersion parse(std::string_view version_str) {
        FirmwareVersion version;
        std::istringstream iss{std::string(version_str)};
        std::string token;
        
        if (std::getline(iss, token, '.')) version.major = std::stoul(token);
        if (std::getline(iss, token, '.')) version.minor = std::stoul(token);
        if (std::getline(iss, token, '.')) version.patch = std::stoul(token);
        
        return version;
    }
};

/**
 * Represents a firmware package with metadata and content
 * @intuition: Bundle version info with actual firmware data for complete package management
 * @approach: Store version, size, and simulated binary data with integrity checking
 * @complexity: Time O(1) for operations, Space O(size)
 */
struct FirmwarePackage {
    FirmwareVersion version;
    std::vector<uint8_t> data;
    std::string checksum;
    size_t total_size;
    
    FirmwarePackage(FirmwareVersion ver, size_t size) 
        : version(ver), total_size(size) {
        // Simulate firmware binary data
        data.resize(size);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint8_t> dis(0, 255);
        
        std::ranges::generate(data, [&] { return dis(gen); });
        
        // Simple checksum simulation
        checksum = std::format("SHA256:{:08X}", 
            std::accumulate(data.begin(), data.end(), 0u));
    }
    
    [[nodiscard]] bool verify_integrity() const {
        auto calculated = std::format("SHA256:{:08X}",
            std::accumulate(data.begin(), data.end(), 0u));
        return calculated == checksum;
    }
};

/**
 * Manages chunk-based firmware download with failure simulation
 * @intuition: Break large firmware into manageable chunks to simulate real network conditions
 * @approach: Fixed chunk size with progress tracking and realistic failure injection
 * @complexity: Time O(n) where n is number of chunks, Space O(chunk_size)
 */
class ChunkedDownloader {
private:
    static constexpr size_t CHUNK_SIZE = 4096; // 4KB chunks
    std::mt19937 rng_{std::random_device{}()};
    
public:
    struct DownloadResult {
        bool success{false};
        std::vector<uint8_t> chunk_data;
        std::string error_message;
        size_t bytes_downloaded{0};
    };
    
    [[nodiscard]] DownloadResult download_chunk(
        const FirmwarePackage& package, 
        size_t chunk_index,
        double failure_rate = 0.1) {
        
        DownloadResult result;
        
        // Simulate network timeout
        std::uniform_real_distribution<> timeout_dis(0.0, 1.0);
        if (timeout_dis(rng_) < failure_rate) {
            result.error_message = "Network timeout";
            return result;
        }
        
        // Simulate download delay
        std::uniform_int_distribution<> delay_dis(50, 200);
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_dis(rng_)));
        
        size_t start_offset = chunk_index * CHUNK_SIZE;
        size_t chunk_size = std::min(CHUNK_SIZE, package.data.size() - start_offset);
        
        if (start_offset >= package.data.size()) {
            result.error_message = "Chunk index out of range";
            return result;
        }
        
        result.chunk_data.assign(
            package.data.begin() + start_offset,
            package.data.begin() + start_offset + chunk_size
        );
        result.bytes_downloaded = chunk_size;
        result.success = true;
        
        return result;
    }
    
    [[nodiscard]] static size_t calculate_total_chunks(size_t total_size) {
        return (total_size + CHUNK_SIZE - 1) / CHUNK_SIZE;
    }
};

/**
 * Device types with different characteristics and update behaviors
 * @intuition: Different IoT devices have varying capabilities and update requirements
 * @approach: Enum-based device types with associated configuration parameters
 * @complexity: Time O(1), Space O(1)
 */
enum class DeviceType {
    SENSOR_LOW_POWER,
    GATEWAY_HIGH_PERFORMANCE,
    ACTUATOR_REAL_TIME
};

/**
 * Update workflow states for comprehensive status tracking
 * @intuition: Clear state machine for update process monitoring and error handling
 * @approach: Comprehensive state enumeration covering all possible update scenarios
 * @complexity: Time O(1), Space O(1)
 */
enum class UpdateState {
    IDLE,
    QUEUED,
    DOWNLOADING,
    VERIFYING,
    INSTALLING,
    REBOOTING,
    SUCCESS,
    FAILED,
    ROLLED_BACK,
    RECOVERY_MODE
};

/**
 * Complete device representation with update capabilities
 * @intuition: Model real IoT devices with unique characteristics and update history
 * @approach: Device-specific parameters with update state management and logging
 * @complexity: Time O(1) for most operations, Space O(log_entries)
 */
class Device {
private:
    static inline std::atomic<uint32_t> next_id_{1000};
    mutable std::mutex state_mutex_;
    std::vector<std::string> update_log_;
    
public:
    const uint32_t device_id;
    const DeviceType type;
    const std::string name;
    const double update_speed_factor; // Speed multiplier for this device type
    
    std::atomic<UpdateState> current_state{UpdateState::IDLE};
    FirmwareVersion current_version;
    FirmwareVersion previous_version; // For rollback
    std::unique_ptr<FirmwarePackage> staged_firmware;
    
    Device(DeviceType device_type, std::string device_name, FirmwareVersion initial_version)
        : device_id(next_id_++), type(device_type), name(std::move(device_name)),
          update_speed_factor(get_speed_factor(device_type)),
          current_version(initial_version), previous_version(initial_version) {
        log_event(std::format("Device {} initialized with firmware {}", 
                             name, current_version.to_string()));
    }
    
    void log_event(const std::string& event) {
        std::lock_guard lock(state_mutex_);
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        
        update_log_.push_back(std::format("[{}] {}", 
            std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S"), event));
        
        // Keep only last 100 log entries
        if (update_log_.size() > 100) {
            update_log_.erase(update_log_.begin());
        }
    }
    
    [[nodiscard]] std::vector<std::string> get_recent_logs(size_t count = 10) const {
        std::lock_guard lock(state_mutex_);
        auto start = update_log_.size() > count ? update_log_.end() - count : update_log_.begin();
        return std::vector<std::string>(start, update_log_.end());
    }
    
    [[nodiscard]] bool can_update_to(const FirmwareVersion& target_version) const {
        return target_version > current_version;
    }
    
    void stage_firmware(std::unique_ptr<FirmwarePackage> package) {
        staged_firmware = std::move(package);
        log_event(std::format("Firmware {} staged for installation", 
                             staged_firmware->version.to_string()));
    }
    
    [[nodiscard]] bool install_staged_firmware() {
        if (!staged_firmware || !staged_firmware->verify_integrity()) {
            log_event("Installation failed: Invalid or corrupted staged firmware");
            return false;
        }
        
        previous_version = current_version;
        current_version = staged_firmware->version;
        staged_firmware.reset();
        
        log_event(std::format("Successfully installed firmware {}", 
                             current_version.to_string()));
        return true;
    }
    
    void rollback() {
        if (previous_version == current_version) {
            log_event("Rollback skipped: No previous version available");
            return;
        }
        
        auto temp = current_version;
        current_version = previous_version;
        previous_version = temp;
        
        log_event(std::format("Rolled back to firmware {}", 
                             current_version.to_string()));
    }
    
private:
    [[nodiscard]] static constexpr double get_speed_factor(DeviceType type) {
        switch (type) {
            case DeviceType::SENSOR_LOW_POWER: return 0.5;
            case DeviceType::GATEWAY_HIGH_PERFORMANCE: return 2.0;
            case DeviceType::ACTUATOR_REAL_TIME: return 1.0;
        }
        return 1.0;
    }
};

/**
 * Comprehensive OTA update management system
 * @intuition: Centralized coordinator for all update operations with queue management
 * @approach: Thread-safe update queue with worker thread and CLI monitoring interface
 * @complexity: Time O(n*m) where n=devices, m=chunks per update, Space O(devices + queued_updates)
 */
class OTAManager {
private:
    std::unordered_map<uint32_t, std::unique_ptr<Device>> devices_;
    std::unordered_map<std::string, std::unique_ptr<FirmwarePackage>> firmware_repository_;
    std::queue<uint32_t> update_queue_;
    
    mutable std::mutex manager_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> running_{true};
    std::thread worker_thread_;
    
    ChunkedDownloader downloader_;
    std::mt19937 rng_{std::random_device{}()};
    
public:
    OTAManager() : worker_thread_(&OTAManager::update_worker, this) {}
    
    ~OTAManager() {
        running_ = false;
        queue_cv_.notify_all();
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
    }
    
    void add_device(std::unique_ptr<Device> device) {
        std::lock_guard lock(manager_mutex_);
        auto device_id = device->device_id;
        devices_[device_id] = std::move(device);
        std::cout << std::format("Device {} added to OTA manager\n", device_id);
    }
    
    void add_firmware(std::unique_ptr<FirmwarePackage> package) {
        std::lock_guard lock(manager_mutex_);
        auto version_key = package->version.to_string();
        firmware_repository_[version_key] = std::move(package);
        std::cout << std::format("Firmware {} added to repository\n", version_key);
    }
    
    [[nodiscard]] bool queue_update(uint32_t device_id, const FirmwareVersion& target_version) {
        std::lock_guard lock(manager_mutex_);
        
        auto device_it = devices_.find(device_id);
        if (device_it == devices_.end()) {
            std::cout << std::format("Error: Device {} not found\n", device_id);
            return false;
        }
        
        auto& device = device_it->second;
        if (!device->can_update_to(target_version)) {
            std::cout << std::format("Error: Cannot update device {} to version {}\n", 
                                   device_id, target_version.to_string());
            return false;
        }
        
        auto firmware_key = target_version.to_string();
        if (firmware_repository_.find(firmware_key) == firmware_repository_.end()) {
            std::cout << std::format("Error: Firmware {} not found in repository\n", firmware_key);
            return false;
        }
        
        device->current_state = UpdateState::QUEUED;
        update_queue_.push(device_id);
        queue_cv_.notify_one();
        
        std::cout << std::format("Update queued for device {} to firmware {}\n", 
                                device_id, target_version.to_string());
        return true;
    }
    
    void rollback_device(uint32_t device_id) {
        std::lock_guard lock(manager_mutex_);
        
        auto device_it = devices_.find(device_id);
        if (device_it == devices_.end()) {
            std::cout << std::format("Error: Device {} not found\n", device_id);
            return;
        }
        
        auto& device = device_it->second;
        device->rollback();
        device->current_state = UpdateState::ROLLED_BACK;
        
        std::cout << std::format("Device {} rolled back to firmware {}\n", 
                                device_id, device->current_version.to_string());
    }
    
    [[nodiscard]] std::vector<Device*> list_devices() const {
        std::lock_guard lock(manager_mutex_);
        std::vector<Device*> device_list;
        device_list.reserve(devices_.size());
        
        for (const auto& [id, device] : devices_) {
            device_list.push_back(device.get());
        }
        
        return device_list;
    }
    
    void print_device_status() const {
        std::lock_guard lock(manager_mutex_);
        
        std::cout << "\n=== Device Status ===\n";
        std::cout << std::format("{:<8} {:<20} {:<15} {:<12} {:<10}\n", 
                                "ID", "Name", "Type", "Version", "State");
        std::cout << std::string(70, '-') << "\n";
        
        for (const auto& [id, device] : devices_) {
            std::cout << std::format("{:<8} {:<20} {:<15} {:<12} {:<10}\n",
                device->device_id,
                device->name,
                device_type_to_string(device->type),
                device->current_version.to_string(),
                update_state_to_string(device->current_state.load())
            );
        }
        std::cout << "\n";
    }
    
private:
    void update_worker() {
        while (running_) {
            std::unique_lock lock(manager_mutex_);
            queue_cv_.wait(lock, [this] { return !update_queue_.empty() || !running_; });
            
            if (!running_) break;
            
            auto device_id = update_queue_.front();
            update_queue_.pop();
            
            auto device_it = devices_.find(device_id);
            if (device_it == devices_.end()) {
                continue;
            }
            
            auto& device = device_it->second;
            lock.unlock();
            
            perform_update(*device);
        }
    }
    
    void perform_update(Device& device) {
        device.current_state = UpdateState::DOWNLOADING;
        device.log_event("Starting OTA update");
        
        // Find target firmware (assume highest version for simplicity)
        std::lock_guard lock(manager_mutex_);
        FirmwarePackage* target_firmware = nullptr;
        FirmwareVersion highest_version{0, 0, 0};
        
        for (const auto& [version_str, package] : firmware_repository_) {
            if (package->version > device.current_version && package->version > highest_version) {
                highest_version = package->version;
                target_firmware = package.get();
            }
        }
        
        if (!target_firmware) {
            device.current_state = UpdateState::FAILED;
            device.log_event("No suitable firmware found for update");
            return;
        }
        
        // Simulate chunked download with potential failures
        if (!download_firmware_chunked(device, *target_firmware)) {
            device.current_state = UpdateState::FAILED;
            return;
        }
        
        // Verification phase
        device.current_state = UpdateState::VERIFYING;
        device.log_event("Verifying downloaded firmware");
        
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        if (!device.staged_firmware->verify_integrity()) {
            device.current_state = UpdateState::FAILED;
            device.log_event("Firmware verification failed");
            return;
        }
        
        // Installation phase
        device.current_state = UpdateState::INSTALLING;
        device.log_event("Installing firmware");
        
        // Simulate power failure during installation
        std::uniform_real_distribution<> power_failure_dis(0.0, 1.0);
        if (power_failure_dis(rng_) < 0.05) { // 5% chance of power failure
            device.current_state = UpdateState::RECOVERY_MODE;
            device.log_event("Power failure during installation - entering recovery mode");
            
            // Simulate recovery process
            std::this_thread::sleep_for(std::chrono::seconds(2));
            device.rollback();
            device.current_state = UpdateState::ROLLED_BACK;
            return;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(
            static_cast<int>(1000 / device.update_speed_factor)));
        
        if (!device.install_staged_firmware()) {
            device.current_state = UpdateState::FAILED;
            return;
        }
        
        // Reboot simulation
        device.current_state = UpdateState::REBOOTING;
        device.log_event("Rebooting device");
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        
        device.current_state = UpdateState::SUCCESS;
        device.log_event(std::format("OTA update completed successfully to version {}", 
                                   device.current_version.to_string()));
    }
    
    [[nodiscard]] bool download_firmware_chunked(Device& device, const FirmwarePackage& firmware) {
        device.log_event(std::format("Downloading firmware {} ({} bytes)", 
                                    firmware.version.to_string(), firmware.total_size));
        
        auto total_chunks = ChunkedDownloader::calculate_total_chunks(firmware.total_size);
        std::vector<uint8_t> downloaded_data;
        downloaded_data.reserve(firmware.total_size);
        
        for (size_t chunk_idx = 0; chunk_idx < total_chunks; ++chunk_idx) {
            auto result = downloader_.download_chunk(firmware, chunk_idx, 0.08); // 8% failure rate
            
            if (!result.success) {
                device.log_event(std::format("Download failed at chunk {}: {}", 
                                            chunk_idx, result.error_message));
                
                // Retry logic
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                result = downloader_.download_chunk(firmware, chunk_idx, 0.03); // Lower failure rate on retry
                
                if (!result.success) {
                    device.log_event("Retry failed, aborting download");
                    return false;
                }
            }
            
            downloaded_data.insert(downloaded_data.end(), 
                                 result.chunk_data.begin(), result.chunk_data.end());
            
            // Progress update
            if (chunk_idx % 10 == 0) {
                auto progress = (chunk_idx * 100) / total_chunks;
                device.log_event(std::format("Download progress: {}%", progress));
            }
        }
        
        // Create staged firmware package
        auto staged = std::make_unique<FirmwarePackage>(firmware.version, firmware.total_size);
        staged->data = std::move(downloaded_data);
        staged->checksum = firmware.checksum;
        
        device.stage_firmware(std::move(staged));
        device.log_event("Firmware download completed");
        
        return true;
    }
    
    [[nodiscard]] static constexpr std::string_view device_type_to_string(DeviceType type) {
        switch (type) {
            case DeviceType::SENSOR_LOW_POWER: return "Sensor";
            case DeviceType::GATEWAY_HIGH_PERFORMANCE: return "Gateway";
            case DeviceType::ACTUATOR_REAL_TIME: return "Actuator";
        }
        return "Unknown";
    }
    
    [[nodiscard]] static constexpr std::string_view update_state_to_string(UpdateState state) {
        switch (state) {
            case UpdateState::IDLE: return "Idle";
            case UpdateState::QUEUED: return "Queued";
            case UpdateState::DOWNLOADING: return "Downloading";
            case UpdateState::VERIFYING: return "Verifying";
            case UpdateState::INSTALLING: return "Installing";
            case UpdateState::REBOOTING: return "Rebooting";
            case UpdateState::SUCCESS: return "Success";
            case UpdateState::FAILED: return "Failed";
            case UpdateState::ROLLED_BACK: return "Rolled Back";
            case UpdateState::RECOVERY_MODE: return "Recovery";
        }
        return "Unknown";
    }
};

/**
 * Command-line interface for interactive OTA management
 * @intuition: Provide real-time monitoring and control capabilities for update operations
 * @approach: Menu-driven CLI with comprehensive device management and update control
 * @complexity: Time O(1) per command, Space O(command_history)
 */
class CLI {
private:
    OTAManager& ota_manager_;
    bool running_{true};
    
public:
    explicit CLI(OTAManager& manager) : ota_manager_(manager) {}
    
    void run() {
        std::cout << "=== OTA Update Simulator ===\n";
        std::cout << "Firmware Engineer's IoT Update Management System\n\n";
        
        while (running_) {
            print_menu();
            process_command();
        }
    }
    
private:
    void print_menu() {
        std::cout << "\nAvailable Commands:\n";
        std::cout << "1. List devices\n";
        std::cout << "2. Show device details\n";
        std::cout << "3. Queue update\n";
        std::cout << "4. Rollback device\n";
        std::cout << "5. Monitor updates (live)\n";
        std::cout << "6. Add test devices\n";
        std::cout << "7. Exit\n";
        std::cout << "Enter command (1-7): ";
    }
    
    void process_command() {
        int choice;
        std::cin >> choice;
        
        switch (choice) {
            case 1: ota_manager_.print_device_status(); break;
            case 2: show_device_details(); break;
            case 3: queue_update_interactive(); break;
            case 4: rollback_device_interactive(); break;
            case 5: monitor_updates(); break;
            case 6: add_test_devices(); break;
            case 7: running_ = false; break;
            default: std::cout << "Invalid command. Please try again.\n";
        }
    }
    
    void show_device_details() {
        std::cout << "Enter device ID: ";
        uint32_t device_id;
        std::cin >> device_id;
        
        auto devices = ota_manager_.list_devices();
        auto device_it = std::ranges::find_if(devices, 
            [device_id](const Device* d) { return d->device_id == device_id; });
        
        if (device_it == devices.end()) {
            std::cout << "Device not found.\n";
            return;
        }
        
        const auto& device = **device_it;
        std::cout << std::format("\n=== Device {} Details ===\n", device_id);
        std::cout << std::format("Name: {}\n", device.name);
        std::cout << std::format("Type: {}\n", device_type_to_string(device.type));
        std::cout << std::format("Current Version: {}\n", device.current_version.to_string());
        std::cout << std::format("Previous Version: {}\n", device.previous_version.to_string());
        std::cout << std::format("Current State: {}\n", update_state_to_string(device.current_state.load()));
        
        std::cout << "\nRecent Log Entries:\n";
        auto logs = device.get_recent_logs(5);
        for (const auto& log : logs) {
            std::cout << "  " << log << "\n";
        }
    }
    
    void queue_update_interactive() {
        std::cout << "Enter device ID: ";
        uint32_t device_id;
        std::cin >> device_id;
        
        std::cout << "Enter target version (major.minor.patch): ";
        std::string version_str;
        std::cin >> version_str;
        
        auto target_version = FirmwareVersion::parse(version_str);
        ota_manager_.queue_update(device_id, target_version);
    }
    
    void rollback_device_interactive() {
        std::cout << "Enter device ID to rollback: ";
        uint32_t device_id;
        std::cin >> device_id;
        
        ota_manager_.rollback_device(device_id);
    }
    
    void monitor_updates() {
        std::cout << "Monitoring updates (press Enter to stop)...\n";
        
        auto start_time = std::chrono::steady_clock::now();
        while (true) {
            // Clear screen (simplified)
            std::cout << "\033[2J\033[H";
            
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time);
            
            std::cout << std::format("=== Live Update Monitor ({}s) ===\n", elapsed.count());
            ota_manager_.print_device_status();
            
            // Check for user input (non-blocking simulation)
            std::this_thread::sleep_for(std::chrono::seconds(2));
            
            // Simple break condition (in real implementation, use non-blocking input)
            if (elapsed.count() > 30) {
                std::cout << "Monitor timeout reached. Press Enter to continue...";
                std::cin.ignore();
                std::cin.get();
                break;
            }
        }
    }
    
    void add_test_devices() {
        // Add sample devices and firmware for testing
        auto sensor1 = std::make_unique<Device>(DeviceType::SENSOR_LOW_POWER, 
                                               "Temperature Sensor 001", 
                                               FirmwareVersion{1, 0, 0});
        
        auto gateway1 = std::make_unique<Device>(DeviceType::GATEWAY_HIGH_PERFORMANCE,
                                                "IoT Gateway Alpha",
                                                FirmwareVersion{2, 1, 0});
        
        auto actuator1 = std::make_unique<Device>(DeviceType::ACTUATOR_REAL_TIME,
                                                 "Valve Controller 42",
                                                 FirmwareVersion{1, 5, 2});
        
        ota_manager_.add_device(std::move(sensor1));
        ota_manager_.add_device(std::move(gateway1));
        ota_manager_.add_device(std::move(actuator1));
        
        // Add test firmware versions
        auto firmware_v1_1_0 = std::make_unique<FirmwarePackage>(FirmwareVersion{1, 1, 0}, 65536);
        auto firmware_v2_0_0 = std::make_unique<FirmwarePackage>(FirmwareVersion{2, 0, 0}, 131072);
        auto firmware_v2_2_0 = std::make_unique<FirmwarePackage>(FirmwareVersion{2, 2, 0}, 98304);
        
        ota_manager_.add_firmware(std::move(firmware_v1_1_0));
        ota_manager_.add_firmware(std::move(firmware_v2_0_0));
        ota_manager_.add_firmware(std::move(firmware_v2_2_0));
        
        std::cout << "Test devices and firmware added successfully!\n";
    }
    
    [[nodiscard]] static constexpr std::string_view device_type_to_string(DeviceType type) {
        switch (type) {
            case DeviceType::SENSOR_LOW_POWER: return "Low Power Sensor";
            case DeviceType::GATEWAY_HIGH_PERFORMANCE: return "High Performance Gateway";
            case DeviceType::ACTUATOR_REAL_TIME: return "Real-Time Actuator";
        }
        return "Unknown Device Type";
    }
    
    [[nodiscard]] static constexpr std::string_view update_state_to_string(UpdateState state) {
        switch (state) {
            case UpdateState::IDLE: return "Idle";
            case UpdateState::QUEUED: return "Queued";
            case UpdateState::DOWNLOADING: return "Downloading";
            case UpdateState::VERIFYING: return "Verifying";
            case UpdateState::INSTALLING: return "Installing";
            case UpdateState::REBOOTING: return "Rebooting";
            case UpdateState::SUCCESS: return "Success";
            case UpdateState::FAILED: return "Failed";
            case UpdateState::ROLLED_BACK: return "Rolled Back";
            case UpdateState::RECOVERY_MODE: return "Recovery Mode";
        }
        return "Unknown State";
    }
};

} // namespace ota_simulator

/**
 * Application entry point with complete OTA simulation setup
 * @intuition: Initialize system components and provide interactive experience
 * @approach: Create manager, populate with test data, and launch CLI interface
 * @complexity: Time O(1) initialization, Space O(devices + firmware_packages)
 */
int main() {
    using namespace ota_simulator;
    
    try {
        OTAManager ota_manager;
        CLI cli(ota_manager);
        
        std::cout << "Initializing OTA Update Simulator for IoT Firmware Management...\n";
        std::cout << "System supports:\n";
        std::cout << "- Multiple device types with varying capabilities\n";
        std::cout << "- Chunked downloads with bandwidth optimization\n";
        std::cout << "- Network interruption and power failure simulation\n";
        std::cout << "- Automatic rollback mechanisms\n";
        std::cout << "- Comprehensive logging and monitoring\n\n";
        
        cli.run();
        
        std::cout << "OTA Update Simulator shutdown complete.\n";
        
    } catch (const std::exception& e) {
        std::cerr << std::format("Fatal error: {}\n", e.what());
        return 1;
    }
    
    return 0;
}
