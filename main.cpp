#include <iostream>
#include <string>
#include <string_view>
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
#include <numeric>
#include <functional>
#include <limits>

namespace ota_simulator {

// Forward declarations
class Device;
class OTAManager;
enum class UpdateState;

/**
 * Custom transparent hasher for string-like types
 * @intuition: Enable heterogeneous lookups to avoid temporary string allocations
 * @approach: Hash both string and string_view types transparently using same algorithm
 * @complexity: Time O(n) where n is string length, Space O(1)
 */
struct TransparentStringHash {
    using is_transparent = void;
    
    [[nodiscard]] constexpr size_t operator()(std::string_view sv) const noexcept {
        return std::hash<std::string_view>{}(sv);
    }
    
    [[nodiscard]] constexpr size_t operator()(const std::string& s) const noexcept {
        return std::hash<std::string_view>{}(s);
    }
    
    [[nodiscard]] constexpr size_t operator()(const char* s) const noexcept {
        return std::hash<std::string_view>{}(s);
    }
};

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
        std::uniform_int_distribution dis(uint8_t{0}, uint8_t{255}); // Class template argument deduction
        
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
        double failure_rate = 0.1) const { // Made const
        
        DownloadResult result;
        
        // Use init-statement for timeout_dis
        if (std::uniform_real_distribution timeout_dis(0.0, 1.0); 
            timeout_dis(rng_) < failure_rate) {
            result.error_message = "Network timeout";
            return result;
        }
        
        // Simulate download delay - Class template argument deduction
        std::uniform_int_distribution delay_dis(50, 200);
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
    
    [[nodiscard]] static constexpr size_t calculate_total_chunks(size_t total_size) noexcept {
        return total_size == 0 ? 0 : (total_size + CHUNK_SIZE - 1) / CHUNK_SIZE;
    }

private:
    static constexpr size_t CHUNK_SIZE = 4096; // 4KB chunks
    mutable std::mt19937 rng_{std::random_device{}()};
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
public:
    // ALL public data members grouped together first
    const uint32_t device_id{next_id_++};
    const DeviceType type;
    const std::string name;
    const double update_speed_factor;
    std::atomic<UpdateState> current_state{UpdateState::IDLE};
    FirmwareVersion current_version;
    FirmwareVersion previous_version;
    std::unique_ptr<FirmwarePackage> staged_firmware;
    
    // ALL public methods grouped together
    Device(DeviceType device_type, std::string device_name, FirmwareVersion initial_version)
        : type(device_type), name(std::move(device_name)),
          update_speed_factor(get_speed_factor(device_type)),
          current_version(initial_version), previous_version(initial_version) {
        log_event(std::format("Device {} initialized with firmware {}", 
                             name, current_version.to_string()));
    }
    
    void log_event(const std::string& event) {
        std::lock_guard lock(state_mutex_);
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        
        struct tm time_buf{};
        localtime_r(&time_t, &time_buf);
        
        update_log_.push_back(std::format("[{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}] {}", 
            time_buf.tm_year + 1900, time_buf.tm_mon + 1, time_buf.tm_mday,
            time_buf.tm_hour, time_buf.tm_min, time_buf.tm_sec, event));
        
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
    // ALL private data members grouped together
    static inline std::atomic<uint32_t> next_id_{1000};
    mutable std::mutex state_mutex_;
    std::vector<std::string> update_log_;
    
    // ALL private methods grouped together
    [[nodiscard]] static constexpr double get_speed_factor(DeviceType type) {
        using enum DeviceType;
        switch (type) {
            case SENSOR_LOW_POWER: return 0.5;
            case GATEWAY_HIGH_PERFORMANCE: return 2.0;
            case ACTUATOR_REAL_TIME: return 1.0;
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
public:
    OTAManager() : worker_thread_(&OTAManager::update_worker, this) {}
    
    ~OTAManager() {
        running_ = false;
        queue_cv_.notify_all();
        // std::jthread automatically joins on destruction
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
        
        const auto& device = device_it->second; // const reference
        if (!device->can_update_to(target_version)) {
            std::cout << std::format("Error: Cannot update device {} to version {}\n", 
                                   device_id, target_version.to_string());
            return false;
        }
        
        // Use init-statement and contains()
        if (auto firmware_key = target_version.to_string(); 
            !firmware_repository_.contains(firmware_key)) {
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
        
        const auto& device = device_it->second; // const reference
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
    std::unordered_map<uint32_t, std::unique_ptr<Device>> devices_;
    std::unordered_map<std::string, std::unique_ptr<FirmwarePackage>, 
                      TransparentStringHash, std::equal_to<>> firmware_repository_; // Transparent hasher
    std::queue<uint32_t> update_queue_;
    
    mutable std::mutex manager_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> running_{true};
    std::jthread worker_thread_; // Use std::jthread instead of std::thread
    
    ChunkedDownloader downloader_;
    mutable std::mt19937 rng_{std::random_device{}()};
    
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
            
            const auto& device = device_it->second; // const reference
            lock.unlock();
            
            perform_update(*device);
        }
    }
    
    void perform_update(Device& device) {
        using enum UpdateState; // Reduce verbosity
        
        device.current_state = DOWNLOADING;
        device.log_event("Starting OTA update");
        
        // Find target firmware (assume highest version for simplicity)
        std::lock_guard lock(manager_mutex_);
        const FirmwarePackage* target_firmware = nullptr; // pointer-to-const
        FirmwareVersion highest_version{0, 0, 0};
        
        for (const auto& [version_str, package] : firmware_repository_) {
            if (package->version > device.current_version && package->version > highest_version) {
                highest_version = package->version;
                target_firmware = package.get();
            }
        }
        
        if (!target_firmware) {
            device.current_state = FAILED;
            device.log_event("No suitable firmware found for update");
            return;
        }
        
        // Simulate chunked download with potential failures
        if (!download_firmware_chunked(device, *target_firmware)) {
            device.current_state = FAILED;
            return;
        }
        
        // Verification phase
        device.current_state = VERIFYING;
        device.log_event("Verifying downloaded firmware");
        
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        if (!device.staged_firmware->verify_integrity()) {
            device.current_state = FAILED;
            device.log_event("Firmware verification failed");
            return;
        }
        
        // Installation phase
        device.current_state = INSTALLING;
        device.log_event("Installing firmware");
        
        // Use init-statement for power_failure_dis
        if (std::uniform_real_distribution power_failure_dis(0.0, 1.0);
            power_failure_dis(rng_) < 0.05) { // 5% chance of power failure
            device.current_state = RECOVERY_MODE;
            device.log_event("Power failure during installation - entering recovery mode");
            
            // Simulate recovery process
            std::this_thread::sleep_for(std::chrono::seconds(2));
            device.rollback();
            device.current_state = ROLLED_BACK;
            return;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(
            static_cast<int>(1000 / device.update_speed_factor)));
        
        if (!device.install_staged_firmware()) {
            device.current_state = FAILED;
            return;
        }
        
        // Reboot simulation
        device.current_state = REBOOTING;
        device.log_event("Rebooting device");
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        
        device.current_state = SUCCESS;
        device.log_event(std::format("OTA update completed successfully to version {}", 
                                   device.current_version.to_string()));
    }
    
    [[nodiscard]] bool download_firmware_chunked(Device& device, const FirmwarePackage& firmware) const { // Made const
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
        using enum DeviceType; // Reduce verbosity
        switch (type) {
            case SENSOR_LOW_POWER: return "Sensor";
            case GATEWAY_HIGH_PERFORMANCE: return "Gateway";
            case ACTUATOR_REAL_TIME: return "Actuator";
        }
        return "Unknown";
    }
    
    [[nodiscard]] static constexpr std::string_view update_state_to_string(UpdateState state) {
        using enum UpdateState; // Reduce verbosity
        switch (state) {
            case IDLE: return "Idle";
            case QUEUED: return "Queued";
            case DOWNLOADING: return "Downloading";
            case VERIFYING: return "Verifying";
            case INSTALLING: return "Installing";
            case REBOOTING: return "Rebooting";
            case SUCCESS: return "Success";
            case FAILED: return "Failed";
            case ROLLED_BACK: return "Rolled Back";
            case RECOVERY_MODE: return "Recovery";
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
    OTAManager& ota_manager_;
    bool running_{true};
    
    void print_menu() const { // Made const
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
        if (!(std::cin >> choice)) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "Invalid input. Please try again.\n";
            return;
        }
        
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
    
    void show_device_details() const { // Made const
        std::cout << "Enter device ID: ";
        uint32_t device_id;
        if (!(std::cin >> device_id)) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "Invalid device ID.\n";
            return;
        }
        
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
    
    void queue_update_interactive() const { // Made const
        std::cout << "Enter device ID: ";
        uint32_t device_id;
        if (!(std::cin >> device_id)) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "Invalid device ID.\n";
            return;
        }
        
        std::cout << "Enter target version (major.minor.patch): ";
        std::string version_str;
        if (!(std::cin >> version_str)) {
            std::cout << "Invalid version format.\n";
            return;
        }
        
        auto target_version = FirmwareVersion::parse(version_str);
        [[maybe_unused]] auto result = ota_manager_.queue_update(device_id, target_version);
    }
    
    void rollback_device_interactive() const { // Made const
        std::cout << "Enter device ID to rollback: ";
        uint32_t device_id;
        if (!(std::cin >> device_id)) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "Invalid device ID.\n";
            return;
        }
        
        ota_manager_.rollback_device(device_id);
    }
    
    void monitor_updates() const { // Made const
        std::cout << "Monitoring updates (press Enter to stop)...\n";
        
        auto start_time = std::chrono::steady_clock::now();
        while (true) {
            // Clear screen using proper bounded syntax for escape sequences
            std::cout << "\x1B[2J\x1B[H"; // Fixed: bounded syntax with uppercase B
            
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
    
    void add_test_devices() const { // Made const
        using enum DeviceType; // Reduce verbosity
        
        // Add sample devices and firmware for testing
        auto sensor1 = std::make_unique<Device>(SENSOR_LOW_POWER, 
                                               "Temperature Sensor 001", 
                                               FirmwareVersion{1, 0, 0});
        
        auto gateway1 = std::make_unique<Device>(GATEWAY_HIGH_PERFORMANCE,
                                                "IoT Gateway Alpha",
                                                FirmwareVersion{2, 1, 0});
        
        auto actuator1 = std::make_unique<Device>(ACTUATOR_REAL_TIME,
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
        using enum DeviceType; // Reduce verbosity
        switch (type) {
            case SENSOR_LOW_POWER: return "Low Power Sensor";
            case GATEWAY_HIGH_PERFORMANCE: return "High Performance Gateway";
            case ACTUATOR_REAL_TIME: return "Real-Time Actuator";
        }
        return "Unknown Device Type";
    }
    
    [[nodiscard]] static constexpr std::string_view update_state_to_string(UpdateState state) {
        using enum UpdateState; // Reduce verbosity
        switch (state) {
            case IDLE: return "Idle";
            case QUEUED: return "Queued";
            case DOWNLOADING: return "Downloading";
            case VERIFYING: return "Verifying";
            case INSTALLING: return "Installing";
            case REBOOTING: return "Rebooting";
            case SUCCESS: return "Success";
            case FAILED: return "Failed";
            case ROLLED_BACK: return "Rolled Back";
            case RECOVERY_MODE: return "Recovery Mode";
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
