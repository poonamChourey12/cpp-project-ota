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
#include <stdexcept>
#include <numeric>
#include <functional>

namespace ota_simulator {

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
 * Logging system with levels for structured debugging
 * @intuition: Provide consistent logging interface with configurable verbosity
 * @approach: Template-based formatting with compile-time level checking
 * @complexity: Time O(1) per log call, Space O(message_size)
 */
enum class LogLevel { DEBUG, INFO, WARNING, ERROR };

class Logger {
public:
    template<typename... Args>
    void log(LogLevel level, std::format_string<Args...> fmt, Args&&... args) const {
        if (level < min_level_) return;
        
        try {
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            
            struct tm time_buf{};
            localtime_r(&time_t, &time_buf);
            
            std::cout << std::format("[{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}] [{}] {}\n",
                time_buf.tm_year + 1900, time_buf.tm_mon + 1, time_buf.tm_mday,
                time_buf.tm_hour, time_buf.tm_min, time_buf.tm_sec,
                level_to_string(level),
                std::format(fmt, std::forward<Args>(args)...));
        } catch (const std::exception& e) {
            std::cerr << "Logging error: " << e.what() << "\n";
        }
    }
    
    void set_level(LogLevel level) noexcept { min_level_ = level; }

private:
    std::atomic<LogLevel> min_level_{LogLevel::INFO};
    
    [[nodiscard]] static constexpr std::string_view level_to_string(LogLevel level) noexcept {
        switch (level) {
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::INFO: return "INFO";
            case LogLevel::WARNING: return "WARN";
            case LogLevel::ERROR: return "ERROR";
        }
        return "UNKNOWN";
    }
};

// Global logger instance
inline Logger g_logger;

/**
 * Result type for operations that can fail
 * @intuition: Provide clear success/failure indication with error details
 * @approach: Use enum for error types with descriptive messages
 * @complexity: Time O(1), Space O(message_size)
 */
enum class OTAError {
    SUCCESS,
    DEVICE_NOT_FOUND,
    INVALID_FIRMWARE,
    NETWORK_ERROR,
    VERIFICATION_FAILED,
    INSTALLATION_FAILED,
    INVALID_INPUT
};

struct OTAResult {
    OTAError error{OTAError::SUCCESS};
    std::string message;
    
    [[nodiscard]] constexpr bool is_success() const noexcept { 
        return error == OTAError::SUCCESS; 
    }
    
    [[nodiscard]] explicit constexpr operator bool() const noexcept { 
        return is_success(); 
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
    
    constexpr auto operator<=>(const FirmwareVersion& other) const noexcept = default;
    
    [[nodiscard]] std::string to_string() const {
        return std::format("{}.{}.{}", major, minor, patch);
    }
    
    [[nodiscard]] static FirmwareVersion parse(std::string_view version_str) {
        if (version_str.empty()) {
            throw std::invalid_argument("Version string cannot be empty");
        }
        
        FirmwareVersion version;
        std::istringstream iss{std::string(version_str)};
        std::string token;
        
        try {
            if (std::getline(iss, token, '.')) version.major = std::stoul(token);
            if (std::getline(iss, token, '.')) version.minor = std::stoul(token);
            if (std::getline(iss, token, '.')) version.patch = std::stoul(token);
        } catch (const std::exception& e) {
            throw std::invalid_argument(std::format("Invalid version format: {}", version_str));
        }
        
        return version;
    }
    
    [[nodiscard]] constexpr bool is_valid() const noexcept {
        return major > 0 || minor > 0 || patch > 0;
    }
};

/**
 * Represents a firmware package with metadata and content
 * @intuition: Bundle version info with actual firmware data for complete package management
 * @approach: Store version, size, and simulated binary data with integrity checking
 * @complexity: Time O(1) for operations, Space O(size)
 */
class FirmwarePackage {
public:
    FirmwareVersion version;
    std::vector<uint8_t> data;
    std::string checksum;
    size_t total_size;
    
    FirmwarePackage(FirmwareVersion ver, size_t size) 
        : version(ver), total_size(size) {
        if (size == 0) {
            throw std::invalid_argument("Firmware size cannot be zero");
        }
        
        if (size > MAX_FIRMWARE_SIZE) {
            throw std::invalid_argument(std::format("Firmware size {} exceeds maximum {}", 
                                                   size, MAX_FIRMWARE_SIZE));
        }
        
        generate_firmware_data(size);
    }
    
    [[nodiscard]] bool verify_integrity() const noexcept {
        try {
            if (data.empty()) {
                return false;
            }
            
            auto calculated = std::format("SHA256:{:08X}",
                std::accumulate(data.begin(), data.end(), 0u));
            return calculated == checksum;
        } catch (...) {
            return false;
        }
    }

private:
    static constexpr size_t MAX_FIRMWARE_SIZE = 10 * 1024 * 1024; // 10MB max
    
    void generate_firmware_data(size_t size) {
        data.resize(size);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint8_t> dis(0, 255);
        
        std::ranges::generate(data, [&] { return dis(gen); });
        
        checksum = std::format("SHA256:{:08X}", 
            std::accumulate(data.begin(), data.end(), 0u));
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
        double failure_rate = 0.1) noexcept {
        
        DownloadResult result;
        
        try {
            if (failure_rate < 0.0 || failure_rate > 1.0) {
                result.error_message = "Invalid failure rate";
                return result;
            }
            
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
            if (start_offset >= package.data.size()) {
                result.error_message = "Chunk index out of range";
                return result;
            }
            
            size_t chunk_size = std::min(CHUNK_SIZE, package.data.size() - start_offset);
            
            result.chunk_data.assign(
                package.data.begin() + start_offset,
                package.data.begin() + start_offset + chunk_size
            );
            result.bytes_downloaded = chunk_size;
            result.success = true;
            
        } catch (const std::exception& e) {
            result.error_message = std::format("Download error: {}", e.what());
        }
        
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
 * Complete device representation with update capabilities
 * @intuition: Model real IoT devices with unique characteristics and update history
 * @approach: Device-specific parameters with update state management and logging
 * @complexity: Time O(1) for most operations, Space O(log_entries)
 */
class Device {
public:
    const uint32_t device_id{next_id_++};
    const DeviceType type;
    const std::string name;
    const double update_speed_factor;
    
    std::atomic<UpdateState> current_state{UpdateState::IDLE};
    FirmwareVersion current_version;
    FirmwareVersion previous_version;
    std::unique_ptr<FirmwarePackage> staged_firmware;
    
    Device(DeviceType device_type, std::string device_name, FirmwareVersion initial_version)
        : type(device_type), name(validate_name(std::move(device_name))),
          update_speed_factor(get_speed_factor(device_type)),
          current_version(initial_version), previous_version(initial_version) {
        
        if (!initial_version.is_valid()) {
            throw std::invalid_argument("Initial firmware version must be valid");
        }
        
        log_event(std::format("Device {} initialized with firmware {}", 
                             name, current_version.to_string()));
    }
    
    void log_event(std::string_view event) {
        if (event.empty()) return;
        
        try {
            std::lock_guard lock(state_mutex_);
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            
            struct tm time_buf{};
            localtime_r(&time_t, &time_buf);
            
            update_log_.emplace_back(std::format("[{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}] {}",
                time_buf.tm_year + 1900, time_buf.tm_mon + 1, time_buf.tm_mday,
                time_buf.tm_hour, time_buf.tm_min, time_buf.tm_sec, event));
            
            // Keep only last 100 log entries
            if (update_log_.size() > MAX_LOG_ENTRIES) {
                update_log_.erase(update_log_.begin());
            }
        } catch (const std::exception& e) {
            g_logger.log(LogLevel::ERROR, "Failed to log event for device {}: {}", device_id, e.what());
        }
    }
    
    [[nodiscard]] std::vector<std::string> get_recent_logs(size_t count = 10) const {
        std::lock_guard lock(state_mutex_);
        
        if (count == 0 || update_log_.empty()) {
            return {};
        }
        
        size_t start_idx = update_log_.size() > count ? update_log_.size() - count : 0;
        return std::vector<std::string>(update_log_.begin() + start_idx, update_log_.end());
    }
    
    [[nodiscard]] bool can_update_to(const FirmwareVersion& target_version) const noexcept {
        return target_version.is_valid() && target_version > current_version;
    }
    
    [[nodiscard]] OTAResult stage_firmware(std::unique_ptr<FirmwarePackage> package) {
        if (!package) {
            return {OTAError::INVALID_FIRMWARE, "Firmware package is null"};
        }
        
        if (!package->verify_integrity()) {
            return {OTAError::VERIFICATION_FAILED, "Firmware integrity check failed"};
        }
        
        staged_firmware = std::move(package);
        log_event(std::format("Firmware {} staged for installation", 
                             staged_firmware->version.to_string()));
        
        return {OTAError::SUCCESS, "Firmware staged successfully"};
    }
    
    [[nodiscard]] OTAResult install_staged_firmware() {
        if (!staged_firmware) {
            return {OTAError::INVALID_FIRMWARE, "No staged firmware available"};
        }
        
        if (!staged_firmware->verify_integrity()) {
            log_event("Installation failed: Staged firmware integrity check failed");
            return {OTAError::VERIFICATION_FAILED, "Staged firmware integrity check failed"};
        }
        
        previous_version = current_version;
        current_version = staged_firmware->version;
        staged_firmware.reset();
        
        log_event(std::format("Successfully installed firmware {}", 
                             current_version.to_string()));
        
        return {OTAError::SUCCESS, "Firmware installed successfully"};
    }
    
    void rollback() noexcept {
        try {
            if (previous_version == current_version) {
                log_event("Rollback skipped: No previous version available");
                return;
            }
            
            auto temp = current_version;
            current_version = previous_version;
            previous_version = temp;
            
            log_event(std::format("Rolled back to firmware {}", 
                                 current_version.to_string()));
        } catch (const std::exception& e) {
            g_logger.log(LogLevel::ERROR, "Rollback failed for device {}: {}", device_id, e.what());
        }
    }
    
    [[nodiscard]] bool has_staged_firmware() const noexcept {
        return staged_firmware != nullptr;
    }

private:
    static inline std::atomic<uint32_t> next_id_{1000};
    static constexpr size_t MAX_LOG_ENTRIES = 100;
    static constexpr size_t MAX_NAME_LENGTH = 100;
    
    mutable std::mutex state_mutex_;
    std::vector<std::string> update_log_;
    
    [[nodiscard]] static std::string validate_name(std::string name) {
        if (name.empty()) {
            throw std::invalid_argument("Device name cannot be empty");
        }
        
        if (name.length() > MAX_NAME_LENGTH) {
            throw std::invalid_argument(std::format("Device name too long (max {} characters)", MAX_NAME_LENGTH));
        }
        
        return name;
    }
    
    [[nodiscard]] static constexpr double get_speed_factor(DeviceType type) noexcept {
        switch (type) {
            case DeviceType::SENSOR_LOW_POWER: return 0.5;
            case DeviceType::GATEWAY_HIGH_PERFORMANCE: return 2.0;
            case DeviceType::ACTUATOR_REAL_TIME: return 1.0;
        }
        return 1.0;
    }
};

/**
 * RAII thread manager for safe thread lifecycle management
 * @intuition: Ensure threads are always properly joined without exceptions
 * @approach: RAII wrapper with noexcept destructor and proper cleanup
 * @complexity: Time O(1), Space O(1)
 */
class ThreadManager {
public:
    ThreadManager(std::thread&& t, std::atomic<bool>& flag) 
        : thread_(std::move(t)), running_flag_(flag) {}
    
    ~ThreadManager() noexcept {
        try {
            running_flag_ = false;
            if (thread_.joinable()) {
                thread_.join();
            }
        } catch (...) {
            // Cannot throw from destructor
        }
    }
    
    ThreadManager(const ThreadManager&) = delete;
    ThreadManager& operator=(const ThreadManager&) = delete;
    ThreadManager(ThreadManager&&) = default;
    ThreadManager& operator=(ThreadManager&&) = default;

private:
    std::thread thread_;
    std::atomic<bool>& running_flag_;
};

/**
 * Comprehensive OTA update management system
 * @intuition: Centralized coordinator for all update operations with queue management
 * @approach: Thread-safe update queue with worker thread and CLI monitoring interface
 * @complexity: Time O(n*m) where n=devices, m=chunks per update, Space O(devices + queued_updates)
 */
class OTAManager {
public:
    OTAManager() : worker_thread_(std::thread(&OTAManager::update_worker, this), running_) {
        g_logger.log(LogLevel::INFO, "OTA Manager initialized");
    }
    
    ~OTAManager() noexcept {
        try {
            running_ = false;
            queue_cv_.notify_all();
            g_logger.log(LogLevel::INFO, "OTA Manager shutting down");
        } catch (...) {
            // Destructor cannot throw
        }
    }
    
    [[nodiscard]] OTAResult add_device(std::unique_ptr<Device> device) {
        if (!device) {
            return {OTAError::INVALID_INPUT, "Device cannot be null"};
        }
        
        try {
            std::lock_guard lock(manager_mutex_);
            auto device_id = device->device_id;
            
            if (devices_.find(device_id) != devices_.end()) {
                return {OTAError::INVALID_INPUT, 
                       std::format("Device {} already exists", device_id)};
            }
            
            devices_[device_id] = std::move(device);
            g_logger.log(LogLevel::INFO, "Device {} added to OTA manager", device_id);
            
            return {OTAError::SUCCESS, "Device added successfully"};
        } catch (const std::exception& e) {
            return {OTAError::INVALID_INPUT, 
                   std::format("Failed to add device: {}", e.what())};
        }
    }
    
    [[nodiscard]] OTAResult add_firmware(std::unique_ptr<FirmwarePackage> package) {
        if (!package) {
            return {OTAError::INVALID_FIRMWARE, "Firmware package cannot be null"};
        }
        
        if (!package->verify_integrity()) {
            return {OTAError::VERIFICATION_FAILED, "Firmware integrity check failed"};
        }
        
        try {
            std::lock_guard lock(manager_mutex_);
            auto version_key = package->version.to_string();
            
            firmware_repository_[std::move(version_key)] = std::move(package);
            g_logger.log(LogLevel::INFO, "Firmware {} added to repository", version_key);
            
            return {OTAError::SUCCESS, "Firmware added successfully"};
        } catch (const std::exception& e) {
            return {OTAError::INVALID_FIRMWARE, 
                   std::format("Failed to add firmware: {}", e.what())};
        }
    }
    
    [[nodiscard]] OTAResult queue_update(uint32_t device_id, const FirmwareVersion& target_version) {
        if (device_id == 0) {
            return {OTAError::INVALID_INPUT, "Invalid device ID"};
        }
        
        if (!target_version.is_valid()) {
            return {OTAError::INVALID_FIRMWARE, "Invalid target version"};
        }
        
        std::lock_guard lock(manager_mutex_);
        
        auto device_it = devices_.find(device_id);
        if (device_it == devices_.end()) {
            return {OTAError::DEVICE_NOT_FOUND, 
                   std::format("Device {} not found", device_id)};
        }
        
        auto& device = device_it->second;
        if (!device->can_update_to(target_version)) {
            return {OTAError::INVALID_FIRMWARE, 
                   std::format("Cannot update device {} to version {}", 
                              device_id, target_version.to_string())};
        }
        
        auto firmware_key = target_version.to_string();
        if (firmware_repository_.find(firmware_key) == firmware_repository_.end()) {
            return {OTAError::INVALID_FIRMWARE, 
                   std::format("Firmware {} not found in repository", firmware_key)};
        }
        
        device->current_state = UpdateState::QUEUED;
        update_queue_.push(device_id);
        queue_cv_.notify_one();
        
        g_logger.log(LogLevel::INFO, "Update queued for device {} to firmware {}", 
                    device_id, target_version.to_string());
        
        return {OTAError::SUCCESS, "Update queued successfully"};
    }
    
    [[nodiscard]] OTAResult rollback_device(uint32_t device_id) {
        if (device_id == 0) {
            return {OTAError::INVALID_INPUT, "Invalid device ID"};
        }
        
        std::lock_guard lock(manager_mutex_);
        
        auto device_it = devices_.find(device_id);
        if (device_it == devices_.end()) {
            return {OTAError::DEVICE_NOT_FOUND, 
                   std::format("Device {} not found", device_id)};
        }
        
        auto& device = device_it->second;
        device->rollback();
        device->current_state = UpdateState::ROLLED_BACK;
        
        g_logger.log(LogLevel::INFO, "Device {} rolled back to firmware {}", 
                    device_id, device->current_version.to_string());
        
        return {OTAError::SUCCESS, "Device rolled back successfully"};
    }
    
    [[nodiscard]] std::vector<const Device*> list_devices() const {
        std::lock_guard lock(manager_mutex_);
        std::vector<const Device*> device_list;
        device_list.reserve(devices_.size());
        
        for (const auto& [id, device] : devices_) {
            device_list.push_back(device.get());
        }
        
        return device_list;
    }
    
    [[nodiscard]] size_t get_device_count() const noexcept {
        std::lock_guard lock(manager_mutex_);
        return devices_.size();
    }
    
    [[nodiscard]] size_t get_queue_size() const noexcept {
        std::lock_guard lock(manager_mutex_);
        return update_queue_.size();
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
                device->name.substr(0, 19), // Truncate long names
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
                      TransparentStringHash, std::equal_to<>> firmware_repository_;
    std::queue<uint32_t> update_queue_;
    
    mutable std::mutex manager_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> running_{true};
    ThreadManager worker_thread_;
    
    ChunkedDownloader downloader_;
    mutable std::mt19937 rng_{std::random_device{}()};
    
    void update_worker() noexcept {
        try {
            while (running_) {
                std::unique_lock lock(manager_mutex_);
                queue_cv_.wait(lock, [this] { return !update_queue_.empty() || !running_; });
                
                if (!running_) break;
                
                if (update_queue_.empty()) continue;
                
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
        } catch (const std::exception& e) {
            g_logger.log(LogLevel::ERROR, "Update worker error: {}", e.what());
        }
    }
    
    void perform_update(Device& device) noexcept {
        try {
            device.current_state = UpdateState::DOWNLOADING;
            device.log_event("Starting OTA update");
            
            // Find target firmware
            std::lock_guard lock(manager_mutex_);
            const FirmwarePackage* target_firmware = find_highest_compatible_firmware(device);
            
            if (!target_firmware) {
                device.current_state = UpdateState::FAILED;
                device.log_event("No suitable firmware found for update");
                return;
            }
            
            // Create a copy for download simulation
            auto firmware_copy = std::make_unique<FirmwarePackage>(
                target_firmware->version, target_firmware->total_size);
            firmware_copy->data = target_firmware->data;
            firmware_copy->checksum = target_firmware->checksum;
            
            lock.~lock_guard(); // Release lock before lengthy download
            
            // Simulate chunked download
            if (!download_firmware_chunked(device, *firmware_copy)) {
                device.current_state = UpdateState::FAILED;
                return;
            }
            
            // Verification phase
            device.current_state = UpdateState::VERIFYING;
            device.log_event("Verifying downloaded firmware");
            
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            
            auto stage_result = device.stage_firmware(std::move(firmware_copy));
            if (!stage_result) {
                device.current_state = UpdateState::FAILED;
                device.log_event(stage_result.message);
                return;
            }
            
            // Installation phase with power failure simulation
            device.current_state = UpdateState::INSTALLING;
            device.log_event("Installing firmware");
            
            if (simulate_power_failure()) {
                device.current_state = UpdateState::RECOVERY_MODE;
                device.log_event("Power failure during installation - entering recovery mode");
                
                std::this_thread::sleep_for(std::chrono::seconds(2));
                device.rollback();
                device.current_state = UpdateState::ROLLED_BACK;
                return;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(
                static_cast<int>(1000 / device.update_speed_factor)));
            
            auto install_result = device.install_staged_firmware();
            if (!install_result) {
                device.current_state = UpdateState::FAILED;
                device.log_event(install_result.message);
                return;
            }
            
            // Reboot simulation
            device.current_state = UpdateState::REBOOTING;
            device.log_event("Rebooting device");
            std::this_thread::sleep_for(std::chrono::milliseconds(2000));
            
            device.current_state = UpdateState::SUCCESS;
            device.log_event(std::format("OTA update completed successfully to version {}", 
                                       device.current_version.to_string()));
            
        } catch (const std::exception& e) {
            device.current_state = UpdateState::FAILED;
            device.log_event(std::format("Update failed with exception: {}", e.what()));
            g_logger.log(LogLevel::ERROR, "Update failed for device {}: {}", device.device_id, e.what());
        }
    }
    
    [[nodiscard]] const FirmwarePackage* find_highest_compatible_firmware(const Device& device) const {
        const FirmwarePackage* best_firmware = nullptr;
        FirmwareVersion highest_version{0, 0, 0};
        
        for (const auto& [version_str, package] : firmware_repository_) {
            if (package->version > device.current_version && package->version > highest_version) {
                highest_version = package->version;
                best_firmware = package.get();
            }
        }
        
        return best_firmware;
    }
    
    [[nodiscard]] bool download_firmware_chunked(Device& device, const FirmwarePackage& firmware) {
        try {
            device.log_event(std::format("Downloading firmware {} ({} bytes)", 
                                        firmware.version.to_string(), firmware.total_size));
            
            auto total_chunks = ChunkedDownloader::calculate_total_chunks(firmware.total_size);
            if (total_chunks == 0) {
                device.log_event("Invalid firmware size for download");
                return false;
            }
            
            std::vector<uint8_t> downloaded_data;
            downloaded_data.reserve(firmware.total_size);
            
            for (size_t chunk_idx = 0; chunk_idx < total_chunks; ++chunk_idx) {
                auto result = downloader_.download_chunk(firmware, chunk_idx, 0.08);
                
                if (!result.success) {
                    device.log_event(std::format("Download failed at chunk {}: {}", 
                                                chunk_idx, result.error_message));
                    
                    // Retry logic
                    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                    result = downloader_.download_chunk(firmware, chunk_idx, 0.03);
                    
                    if (!result.success) {
                        device.log_event("Retry failed, aborting download");
                        return false;
                    }
                }
                
                downloaded_data.insert(downloaded_data.end(), 
                                     result.chunk_data.begin(), result.chunk_data.end());
                
                // Progress update every 10 chunks
                if (chunk_idx % 10 == 0) {
                    auto progress = (chunk_idx * 100) / total_chunks;
                    device.log_event(std::format("Download progress: {}%", progress));
                }
            }
            
            device.log_event("Firmware download completed");
            return true;
            
        } catch (const std::exception& e) {
            device.log_event(std::format("Download failed with exception: {}", e.what()));
            return false;
        }
    }
    
    [[nodiscard]] bool simulate_power_failure() noexcept {
        try {
            std::uniform_real_distribution<> power_failure_dis(0.0, 1.0);
            return power_failure_dis(rng_) < 0.05; // 5% chance
        } catch (...) {
            return false;
        }
    }
    
    [[nodiscard]] static constexpr std::string_view device_type_to_string(DeviceType type) noexcept {
        switch (type) {
            case DeviceType::SENSOR_LOW_POWER: return "Sensor";
            case DeviceType::GATEWAY_HIGH_PERFORMANCE: return "Gateway";
            case DeviceType::ACTUATOR_REAL_TIME: return "Actuator";
        }
        return "Unknown";
    }
    
    [[nodiscard]] static constexpr std::string_view update_state_to_string(UpdateState state) noexcept {
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
public:
    explicit CLI(OTAManager& manager) : ota_manager_(manager) {}
    
    void run() {
        try {
            std::cout << "=== OTA Update Simulator ===\n";
            std::cout << "Firmware Engineer's IoT Update Management System\n\n";
            
            while (running_) {
                print_menu();
                if (!process_command()) {
                    std::cout << "Invalid input. Please try again.\n";
                }
            }
        } catch (const std::exception& e) {
            g_logger.log(LogLevel::ERROR, "CLI error: {}", e.what());
        }
    }

private:
    OTAManager& ota_manager_;
    bool running_{true};
    
    void print_menu() const {
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
    
    [[nodiscard]] bool process_command() {
        int choice;
        if (!(std::cin >> choice)) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            return false;
        }
        
        try {
            switch (choice) {
                case 1: ota_manager_.print_device_status(); break;
                case 2: show_device_details(); break;
                case 3: queue_update_interactive(); break;
                case 4: rollback_device_interactive(); break;
                case 5: monitor_updates(); break;
                case 6: add_test_devices(); break;
                case 7: running_ = false; break;
                default: return false;
            }
        } catch (const std::exception& e) {
            std::cout << std::format("Command failed: {}\n", e.what());
        }
        
        return true;
    }
    
    void show_device_details() const {
        std::cout << "Enter device ID: ";
        uint32_t device_id;
        if (!(std::cin >> device_id)) {
            std::cout << "Invalid device ID format.\n";
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
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
        print_device_info(device);
    }
    
    void print_device_info(const Device& device) const {
        std::cout << std::format("\n=== Device {} Details ===\n", device.device_id);
        std::cout << std::format("Name: {}\n", device.name);
        std::cout << std::format("Type: {}\n", device_type_to_string(device.type));
        std::cout << std::format("Current Version: {}\n", device.current_version.to_string());
        std::cout << std::format("Previous Version: {}\n", device.previous_version.to_string());
        std::cout << std::format("Current State: {}\n", update_state_to_string(device.current_state.load()));
        std::cout << std::format("Has Staged Firmware: {}\n", device.has_staged_firmware() ? "Yes" : "No");
        
        std::cout << "\nRecent Log Entries:\n";
        auto logs = device.get_recent_logs(5);
        for (const auto& log : logs) {
            std::cout << "  " << log << "\n";
        }
    }
    
    void queue_update_interactive() const {
        std::cout << "Enter device ID: ";
        uint32_t device_id;
        if (!(std::cin >> device_id)) {
            std::cout << "Invalid device ID format.\n";
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            return;
        }
        
        std::cout << "Enter target version (major.minor.patch): ";
        std::string version_str;
        if (!(std::cin >> version_str)) {
            std::cout << "Invalid version format.\n";
            return;
        }
        
        try {
            auto target_version = FirmwareVersion::parse(version_str);
            auto result = ota_manager_.queue_update(device_id, target_version);
            
            if (result) {
                std::cout << result.message << "\n";
            } else {
                std::cout << std::format("Failed to queue update: {}\n", result.message);
            }
        } catch (const std::exception& e) {
            std::cout << std::format("Invalid version format: {}\n", e.what());
        }
    }
    
    void rollback_device_interactive() const {
        std::cout << "Enter device ID to rollback: ";
        uint32_t device_id;
        if (!(std::cin >> device_id)) {
            std::cout << "Invalid device ID format.\n";
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            return;
        }
        
        auto result = ota_manager_.rollback_device(device_id);
        if (result) {
            std::cout << result.message << "\n";
        } else {
            std::cout << std::format("Failed to rollback device: {}\n", result.message);
        }
    }
    
    void monitor_updates() const {
        std::cout << "Monitoring updates for 30 seconds...\n";
        std::cout << "Press Ctrl+C to interrupt\n\n";
        
        auto start_time = std::chrono::steady_clock::now();
        constexpr auto monitor_duration = std::chrono::seconds(30);
        
        while (std::chrono::steady_clock::now() - start_time < monitor_duration) {
            // Clear screen (simplified)
            std::cout << "\033[2J\033[H";
            
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start_time);
            
            std::cout << std::format("=== Live Update Monitor ({}s) ===\n", elapsed.count());
            std::cout << std::format("Devices: {} | Queue: {}\n", 
                                   ota_manager_.get_device_count(), 
                                   ota_manager_.get_queue_size());
            ota_manager_.print_device_status();
            
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
        
        std::cout << "Monitor timeout reached. Press Enter to continue...";
        std::cin.ignore();
        std::cin.get();
    }
    
    void add_test_devices() const {
        try {
            // Add sample devices
            auto sensor1 = std::make_unique<Device>(DeviceType::SENSOR_LOW_POWER, 
                                                   "Temperature Sensor 001", 
                                                   FirmwareVersion{1, 0, 0});
            
            auto gateway1 = std::make_unique<Device>(DeviceType::GATEWAY_HIGH_PERFORMANCE,
                                                    "IoT Gateway Alpha",
                                                    FirmwareVersion{2, 1, 0});
            
            auto actuator1 = std::make_unique<Device>(DeviceType::ACTUATOR_REAL_TIME,
                                                     "Valve Controller 42",
                                                     FirmwareVersion{1, 5, 2});
            
            auto results = {
                ota_manager_.add_device(std::move(sensor1)),
                ota_manager_.add_device(std::move(gateway1)),
                ota_manager_.add_device(std::move(actuator1))
            };
            
            // Add test firmware versions
            auto firmware_versions = {
                std::make_unique<FirmwarePackage>(FirmwareVersion{1, 1, 0}, 65536),
                std::make_unique<FirmwarePackage>(FirmwareVersion{2, 0, 0}, 131072),
                std::make_unique<FirmwarePackage>(FirmwareVersion{2, 2, 0}, 98304)
            };
            
            for (auto& firmware : firmware_versions) {
                auto result = ota_manager_.add_firmware(std::move(firmware));
                if (!result) {
                    std::cout << std::format("Failed to add firmware: {}\n", result.message);
                }
            }
            
            bool all_successful = std::ranges::all_of(results, 
                [](const auto& result) { return result.is_success(); });
            
            if (all_successful) {
                std::cout << "Test devices and firmware added successfully!\n";
            } else {
                std::cout << "Some test devices failed to add. Check logs for details.\n";
            }
            
        } catch (const std::exception& e) {
            std::cout << std::format("Failed to add test devices: {}\n", e.what());
        }
    }
    
    [[nodiscard]] static constexpr std::string_view device_type_to_string(DeviceType type) noexcept {
        switch (type) {
            case DeviceType::SENSOR_LOW_POWER: return "Low Power Sensor";
            case DeviceType::GATEWAY_HIGH_PERFORMANCE: return "High Performance Gateway";
            case DeviceType::ACTUATOR_REAL_TIME: return "Real-Time Actuator";
        }
        return "Unknown Device Type";
    }
    
    [[nodiscard]] static constexpr std::string_view update_state_to_string(UpdateState state) noexcept {
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
        // Set logging level
        g_logger.set_level(LogLevel::INFO);
        
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
    } catch (...) {
        std::cerr << "Unknown fatal error occurred\n";
        return 1;
    }
    
    return 0;
}
