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
#include <shared_mutex>
#include <condition_variable>
#include <atomic>
#include <format>
#include <ranges>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cassert>
#include <expected>
#include <variant>

namespace ota_simulator {

// Forward declarations
class Device;
class OTAManager;
enum class UpdateState;
enum class UpdateError;

/**
 * Structured error types for comprehensive error handling
 * @intuition: Type-safe error propagation instead of string-based errors
 * @approach: Enum-based error classification with detailed error context
 * @complexity: Time O(1), Space O(1)
 */
enum class UpdateError {
    NetworkTimeout,
    ChecksumMismatch,
    PowerFailure,
    DeviceNotFound,
    FirmwareNotFound,
    InvalidVersion,
    ConcurrentUpdate,
    InsufficientSpace,
    CorruptedData
};

/**
 * Error context with detailed information for debugging
 * @intuition: Provide comprehensive error information for better debugging
 * @approach: Combine error type with contextual details and retry information
 * @complexity: Time O(1), Space O(message_length)
 */
struct ErrorContext {
    UpdateError error_type;
    std::string message;
    std::chrono::system_clock::time_point timestamp;
    size_t retry_count{0};
    
    ErrorContext(UpdateError type, std::string msg) 
        : error_type(type), message(std::move(msg)), 
          timestamp(std::chrono::system_clock::now()) {}
};

template<typename T>
using Result = std::expected<T, ErrorContext>;

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
    
    /**
     * Convert version to string representation
     * @intuition: Human-readable version format for logging and display
     * @approach: Standard semantic versioning format (major.minor.patch)
     * @complexity: Time O(1), Space O(1)
     */
    [[nodiscard]] std::string to_string() const {
        return std::format("{}.{}.{}", major, minor, patch);
    }
    
    /**
     * Parse version string into structured format
     * @intuition: Convert user input into structured version for comparison
     * @approach: String splitting on dots with numeric conversion
     * @complexity: Time O(n) where n is string length, Space O(1)
     */
    [[nodiscard]] static Result<FirmwareVersion> parse(std::string_view version_str) {
        FirmwareVersion version;
        std::istringstream iss{std::string(version_str)};
        std::string token;
        
        try {
            if (std::getline(iss, token, '.')) version.major = std::stoul(token);
            if (std::getline(iss, token, '.')) version.minor = std::stoul(token);
            if (std::getline(iss, token, '.')) version.patch = std::stoul(token);
            return version;
        } catch (const std::exception& e) {
            return std::unexpected(ErrorContext{UpdateError::InvalidVersion, 
                std::format("Invalid version format: {}", version_str)});
        }
    }
};

/**
 * Represents a firmware package with metadata and content
 * @intuition: Bundle version info with actual firmware data for complete package management
 * @approach: Store version, size, and simulated binary data with integrity checking
 * @complexity: Time O(n) for creation where n is size, Space O(size)
 */
struct FirmwarePackage {
    FirmwareVersion version;
    std::vector<uint8_t> data;
    std::string checksum;
    size_t total_size;
    
    /**
     * Create firmware package with simulated binary data
     * @intuition: Generate realistic firmware data for testing scenarios
     * @approach: Random data generation with checksum calculation
     * @complexity: Time O(size), Space O(size)
     */
    FirmwarePackage(FirmwareVersion ver, size_t size) 
        : version(ver), total_size(size) {
        data.resize(size);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint8_t> dis(0, 255);
        
        std::ranges::generate(data, [&] { return dis(gen); });
        
        // Enhanced checksum with actual hash simulation
        uint32_t hash = std::accumulate(data.begin(), data.end(), 0u,
            [](uint32_t acc, uint8_t byte) { return acc * 31 + byte; });
        checksum = std::format("SHA256:{:08X}", hash);
    }
    
    /**
     * Verify firmware package integrity
     * @intuition: Ensure data hasn't been corrupted during transfer
     * @approach: Recalculate checksum and compare with stored value
     * @complexity: Time O(n) where n is data size, Space O(1)
     */
    [[nodiscard]] Result<bool> verify_integrity() const {
        uint32_t calculated_hash = std::accumulate(data.begin(), data.end(), 0u,
            [](uint32_t acc, uint8_t byte) { return acc * 31 + byte; });
        auto calculated_checksum = std::format("SHA256:{:08X}", calculated_hash);
        
        if (calculated_checksum != checksum) {
            return std::unexpected(ErrorContext{UpdateError::ChecksumMismatch,
                std::format("Checksum mismatch: expected {}, got {}", 
                           checksum, calculated_checksum)});
        }
        return true;
    }
};

/**
 * Manages chunk-based firmware download with failure simulation and recovery
 * @intuition: Break large firmware into manageable chunks to simulate real network conditions
 * @approach: Fixed chunk size with progress tracking, realistic failure injection, and retry logic
 * @complexity: Time O(n) where n is number of chunks, Space O(chunk_size)
 */
class ChunkedDownloader {
private:
    static constexpr size_t CHUNK_SIZE = 4096; // 4KB chunks
    static constexpr size_t MAX_RETRIES = 3;
    std::mt19937 rng_{std::random_device{}()};
    
public:
    /**
     * Result of chunk download operation with comprehensive error context
     * @intuition: Provide detailed download results for error handling and progress tracking
     * @approach: Structured result with success status, data, and error information
     * @complexity: Time O(1), Space O(chunk_size)
     */
    struct DownloadResult {
        bool success{false};
        std::vector<uint8_t> chunk_data;
        std::optional<ErrorContext> error;
        size_t bytes_downloaded{0};
        size_t retry_count{0};
    };
    
    /**
     * Download single firmware chunk with automatic retry
     * @intuition: Reliable chunk download with realistic failure simulation
     * @approach: Attempt download with exponential backoff retry on failure
     * @complexity: Time O(retries), Space O(chunk_size)
     */
    [[nodiscard]] DownloadResult download_chunk(
        const FirmwarePackage& package, 
        size_t chunk_index,
        double failure_rate = 0.08) {
        
        DownloadResult result;
        
        for (size_t retry = 0; retry <= MAX_RETRIES; ++retry) {
            result.retry_count = retry;
            
            // Simulate network conditions
            std::uniform_real_distribution<> failure_dis(0.0, 1.0);
            double current_failure_rate = failure_rate * (1.0 - retry * 0.3); // Reduce failure rate on retry
            
            if (failure_dis(rng_) < current_failure_rate) {
                result.error = ErrorContext{UpdateError::NetworkTimeout,
                    std::format("Network timeout on chunk {} (retry {})", chunk_index, retry)};
                
                if (retry < MAX_RETRIES) {
                    // Exponential backoff
                    auto delay = std::chrono::milliseconds(100 * (1 << retry));
                    std::this_thread::sleep_for(delay);
                    continue;
                }
                return result;
            }
            
            // Simulate realistic download delay
            std::uniform_int_distribution<> delay_dis(50, 200);
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_dis(rng_)));
            
            size_t start_offset = chunk_index * CHUNK_SIZE;
            size_t chunk_size = std::min(CHUNK_SIZE, package.data.size() - start_offset);
            
            if (start_offset >= package.data.size()) {
                result.error = ErrorContext{UpdateError::CorruptedData,
                    "Chunk index out of range"};
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
        
        return result;
    }
    
    /**
     * Calculate total number of chunks for given data size
     * @intuition: Determine download progress tracking requirements
     * @approach: Integer division with ceiling for partial chunks
     * @complexity: Time O(1), Space O(1)
     */
    [[nodiscard]] static constexpr size_t calculate_total_chunks(size_t total_size) noexcept {
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
    RECOVERY_MODE,
    CANCELLED
};

/**
 * Complete device representation with update capabilities and thread safety
 * @intuition: Model real IoT devices with unique characteristics and update history
 * @approach: Device-specific parameters with update state management, logging, and concurrent access protection
 * @complexity: Time O(1) for most operations, Space O(log_entries)
 */
class Device {
private:
    static inline std::atomic<uint32_t> next_id_{1000};
    mutable std::shared_mutex state_mutex_;
    std::vector<std::string> update_log_;
    std::atomic<bool> update_in_progress_{false};
    
public:
    const uint32_t device_id;
    const DeviceType type;
    const std::string name;
    const double update_speed_factor;
    
    std::atomic<UpdateState> current_state{UpdateState::IDLE};
    FirmwareVersion current_version;
    FirmwareVersion previous_version;
    std::unique_ptr<FirmwarePackage> staged_firmware;
    
    /**
     * Initialize device with type-specific characteristics
     * @intuition: Create device with unique identity and capabilities
     * @approach: Auto-increment ID with type-specific performance parameters
     * @complexity: Time O(1), Space O(name_length)
     */
    Device(DeviceType device_type, std::string device_name, FirmwareVersion initial_version)
        : device_id(next_id_++), type(device_type), name(std::move(device_name)),
          update_speed_factor(get_speed_factor(device_type)),
          current_version(initial_version), previous_version(initial_version) {
        log_event(std::format("Device {} initialized with firmware {}", 
                             name, current_version.to_string()));
    }
    
    /**
     * Thread-safe event logging with timestamp
     * @intuition: Maintain audit trail of device operations for debugging
     * @approach: Mutex-protected log with automatic timestamp and size management
     * @complexity: Time O(1) amortized, Space O(log_size)
     */
    void log_event(const std::string& event) {
        std::unique_lock lock(state_mutex_);
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        
        update_log_.push_back(std::format("[{}] {}", 
            std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S"), event));
        
        // Keep only last 100 log entries to prevent memory growth
        if (update_log_.size() > 100) {
            update_log_.erase(update_log_.begin());
        }
    }
    
    /**
     * Retrieve recent log entries for monitoring
     * @intuition: Provide access to device history for debugging and monitoring
     * @approach: Thread-safe copy of recent entries with configurable count
     * @complexity: Time O(min(count, log_size)), Space O(result_size)
     */
    [[nodiscard]] std::vector<std::string> get_recent_logs(size_t count = 10) const {
        std::shared_lock lock(state_mutex_);
        auto start = update_log_.size() > count ? update_log_.end() - count : update_log_.begin();
        return std::vector<std::string>(start, update_log_.end());
    }
    
    /**
     * Check if device can be updated to target version
     * @intuition: Prevent downgrade attempts and concurrent updates
     * @approach: Version comparison with update state validation
     * @complexity: Time O(1), Space O(1)
     */
    [[nodiscard]] Result<bool> can_update_to(const FirmwareVersion& target_version) const {
        if (update_in_progress_.load()) {
            return std::unexpected(ErrorContext{UpdateError::ConcurrentUpdate,
                "Update already in progress for this device"});
        }
        
        if (target_version <= current_version) {
            return std::unexpected(ErrorContext{UpdateError::InvalidVersion,
                std::format("Target version {} is not newer than current {}", 
                           target_version.to_string(), current_version.to_string())});
        }
        
        return true;
    }
    
    /**
     * Stage firmware for installation with validation
     * @intuition: Prepare firmware for installation while validating integrity
     * @approach: Store firmware reference with integrity verification
     * @complexity: Time O(n) where n is firmware size, Space O(1)
     */
    Result<void> stage_firmware(std::unique_ptr<FirmwarePackage> package) {
        if (!package) {
            return std::unexpected(ErrorContext{UpdateError::CorruptedData,
                "Null firmware package provided"});
        }
        
        auto integrity_result = package->verify_integrity();
        if (!integrity_result) {
            return std::unexpected(integrity_result.error());
        }
        
        staged_firmware = std::move(package);
        log_event(std::format("Firmware {} staged for installation", 
                             staged_firmware->version.to_string()));
        return {};
    }
    
    /**
     * Install staged firmware with atomicity guarantees
     * @intuition: Atomically update device firmware with rollback capability
     * @approach: Version swap with previous version preservation
     * @complexity: Time O(1), Space O(1)
     */
    [[nodiscard]] Result<void> install_staged_firmware() {
        if (!staged_firmware) {
            return std::unexpected(ErrorContext{UpdateError::CorruptedData,
                "No staged firmware available for installation"});
        }
        
        auto integrity_result = staged_firmware->verify_integrity();
        if (!integrity_result) {
            return std::unexpected(integrity_result.error());
        }
        
        previous_version = current_version;
        current_version = staged_firmware->version;
        staged_firmware.reset();
        
        log_event(std::format("Successfully installed firmware {}", 
                             current_version.to_string()));
        return {};
    }
    
    /**
     * Rollback to previous firmware version
     * @intuition: Recover from failed updates by reverting to known good state
     * @approach: Atomic version swap with validation
     * @complexity: Time O(1), Space O(1)
     */
    Result<void> rollback() {
        if (previous_version == current_version) {
            return std::unexpected(ErrorContext{UpdateError::InvalidVersion,
                "No previous version available for rollback"});
        }
        
        auto temp = current_version;
        current_version = previous_version;
        previous_version = temp;
        
        log_event(std::format("Rolled back to firmware {}", 
                             current_version.to_string()));
        return {};
    }
    
    /**
     * Set update in progress flag atomically
     * @intuition: Prevent concurrent updates to same device
     * @approach: Atomic flag with compare-and-swap semantics
     * @complexity: Time O(1), Space O(1)
     */
    [[nodiscard]] bool try_lock_for_update() {
        bool expected = false;
        return update_in_progress_.compare_exchange_strong(expected, true);
    }
    
    /**
     * Release update lock
     * @intuition: Allow future updates after current operation completes
     * @approach: Atomic store with release semantics
     * @complexity: Time O(1), Space O(1)
     */
    void unlock_update() noexcept {
        update_in_progress_.store(false);
    }
    
private:
    /**
     * Get device type specific performance multiplier
     * @intuition: Different device types have varying update capabilities
     * @approach: Static mapping from device type to performance factor
     * @complexity: Time O(1), Space O(1)
     */
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
 * Comprehensive OTA update management system with enhanced thread safety
 * @intuition: Centralized coordinator for all update operations with queue management and monitoring
 * @approach: Thread-safe update queue with worker thread, granular locking, and CLI monitoring interface
 * @complexity: Time O(n*m) where n=devices, m=chunks per update, Space O(devices + queued_updates)
 */
class OTAManager {
private:
    std::unordered_map<uint32_t, std::unique_ptr<Device>> devices_;
    std::unordered_map<std::string, std::unique_ptr<FirmwarePackage>> firmware_repository_;
    std::queue<uint32_t> update_queue_;
    
    mutable std::shared_mutex devices_mutex_;
    mutable std::shared_mutex firmware_mutex_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> running_{true};
    std::thread worker_thread_;
    
    ChunkedDownloader downloader_;
    std::mt19937 rng_{std::random_device{}()};
    
public:
    /**
     * Initialize OTA manager with worker thread
     * @intuition: Start background processing for queued updates
     * @approach: Spawn dedicated worker thread for update processing
     * @complexity: Time O(1), Space O(1)
     */
    OTAManager() : worker_thread_(&OTAManager::update_worker, this) {}
    
    /**
     * Cleanup with graceful thread termination
     * @intuition: Ensure clean shutdown of background worker
     * @approach: Signal termination and wait for thread completion
     * @complexity: Time O(pending_updates), Space O(1)
     */
    ~OTAManager() {
        running_ = false;
        queue_cv_.notify_all();
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
    }
    
    /**
     * Add device to management system
     * @intuition: Register device for OTA update capability
     * @approach: Thread-safe insertion with unique ID tracking
     * @complexity: Time O(1), Space O(device_data)
     */
    void add_device(std::unique_ptr<Device> device) {
        std::unique_lock lock(devices_mutex_);
        auto device_id = device->device_id;
        devices_[device_id] = std::move(device);
        std::cout << std::format("Device {} added to OTA manager\n", device_id);
    }
    
    /**
     * Add firmware package to repository
     * @intuition: Make firmware available for device updates
     * @approach: Thread-safe storage with version-based indexing
     * @complexity: Time O(1), Space O(firmware_size)
     */
    void add_firmware(std::unique_ptr<FirmwarePackage> package) {
        std::unique_lock lock(firmware_mutex_);
        auto version_key = package->version.to_string();
        firmware_repository_[version_key] = std::move(package);
        std::cout << std::format("Firmware {} added to repository\n", version_key);
    }
    
    /**
     * Queue device update with comprehensive validation
     * @intuition: Safely schedule update with thorough eligibility checking
     * @approach: Multi-step validation with atomic state transitions
     * @complexity: Time O(1), Space O(1)
     */
    [[nodiscard]] Result<void> queue_update(uint32_t device_id, const FirmwareVersion& target_version) {
        // Read-only device lookup
        std::shared_lock devices_lock(devices_mutex_);
        auto device_it = devices_.find(device_id);
        if (device_it == devices_.end()) {
            return std::unexpected(ErrorContext{UpdateError::DeviceNotFound,
                std::format("Device {} not found", device_id)});
        }
        
        auto& device = device_it->second;
        
        // Check update eligibility
        auto can_update_result = device->can_update_to(target_version);
        if (!can_update_result) {
            return std::unexpected(can_update_result.error());
        }
        
        // Verify firmware availability
        std::shared_lock firmware_lock(firmware_mutex_);
        auto firmware_key = target_version.to_string();
        if (firmware_repository_.find(firmware_key) == firmware_repository_.end()) {
            return std::unexpected(ErrorContext{UpdateError::FirmwareNotFound,
                std::format("Firmware {} not found in repository", firmware_key)});
        }
        
        // Try to lock device for update
        if (!device->try_lock_for_update()) {
            return std::unexpected(ErrorContext{UpdateError::ConcurrentUpdate,
                "Another update is already in progress for this device"});
        }
        
        // Queue the update
        {
            std::unique_lock queue_lock(queue_mutex_);
            device->current_state = UpdateState::QUEUED;
            update_queue_.push(device_id);
        }
        queue_cv_.notify_one();
        
        std::cout << std::format("Update queued for device {} to firmware {}\n", 
                                device_id, target_version.to_string());
        return {};
    }
    
    /**
     * Rollback device to previous firmware version
     * @intuition: Provide manual recovery mechanism for failed updates
     * @approach: Direct device rollback with state management
     * @complexity: Time O(1), Space O(1)
     */
    Result<void> rollback_device(uint32_t device_id) {
        std::shared_lock lock(devices_mutex_);
        
        auto device_it = devices_.find(device_id);
        if (device_it == devices_.end()) {
            return std::unexpected(ErrorContext{UpdateError::DeviceNotFound,
                std::format("Device {} not found", device_id)});
        }
        
        auto& device = device_it->second;
        auto rollback_result = device->rollback();
        if (!rollback_result) {
            return std::unexpected(rollback_result.error());
        }
        
        device->current_state = UpdateState::ROLLED_BACK;
        std::cout << std::format("Device {} rolled back to firmware {}\n", 
                                device_id, device->current_version.to_string());
        return {};
    }
    
    /**
     * Get list of all managed devices
     * @intuition: Provide read-only access to device collection for monitoring
     * @approach: Thread-safe snapshot of device pointers
     * @complexity: Time O(n) where n is device count, Space O(n)
     */
    [[nodiscard]] std::vector<Device*> list_devices() const {
        std::shared_lock lock(devices_mutex_);
        std::vector<Device*> device_list;
        device_list.reserve(devices_.size());
        
        for (const auto& [id, device] : devices_) {
            device_list.push_back(device.get());
        }
        
        return device_list;
    }
    
    /**
     * Display comprehensive device status information
     * @intuition: Provide human-readable system status for monitoring
     * @approach: Formatted table output with current device states
     * @complexity: Time O(n) where n is device count, Space O(1)
     */
    void print_device_status() const {
        std::shared_lock lock(devices_mutex_);
        
        std::cout << "\n=== Device Status ===\n";
        std::cout << std::format("{:<8} {:<20} {:<15} {:<12} {:<15}\n", 
                                "ID", "Name", "Type", "Version", "State");
        std::cout << std::string(75, '-') << "\n";
        
        for (const auto& [id, device] : devices_) {
            std::cout << std::format("{:<8} {:<20} {:<15} {:<12} {:<15}\n",
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
    /**
     * Background worker thread for processing update queue
     * @intuition: Dedicated thread prevents blocking main operations
     * @approach: Event-driven processing with condition variable synchronization
     * @complexity: Time O(∞) running, Space O(1)
     */
    void update_worker() {
        while (running_) {
            std::unique_lock lock(queue_mutex_);
            queue_cv_.wait(lock, [this] { return !update_queue_.empty() || !running_; });
            
            if (!running_) break;
            
            auto device_id = update_queue_.front();
            update_queue_.pop();
            lock.unlock();
            
            // Find device with read lock
            std::shared_lock devices_lock(devices_mutex_);
            auto device_it = devices_.find(device_id);
            if (device_it == devices_.end()) {
                continue;
            }
            
            auto& device = device_it->second;
            devices_lock.unlock();
            
            perform_update(*device);
        }
    }
    
    /**
     * Execute complete update workflow for device
     * @intuition: Orchestrate entire update process with comprehensive error handling
     * @approach: Multi-phase update with state tracking and failure recovery
     * @complexity: Time O(firmware_size / chunk_size), Space O(firmware_size)
     */
    void perform_update(Device& device) {
        // Ensure device is unlocked when function exits
        auto unlock_guard = [&device] { device.unlock_update(); };
        std::unique_ptr<Device, decltype(unlock_guard)> guard(&device, unlock_guard);
        
        device.current_state = UpdateState::DOWNLOADING;
        device.log_event("Starting OTA update");
        
        // Find target firmware
        std::shared_lock firmware_lock(firmware_mutex_);
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
        
        firmware_lock.unlock();
        
        // Download phase
        auto download_result = download_firmware_chunked(device, *target_firmware);
        if (!download_result) {
            device.current_state = UpdateState::FAILED;
            device.log_event(std::format("Download failed: {}", download_result.error().message));
            return;
        }
        
        // Verification phase
        device.current_state = UpdateState::VERIFYING;
        device.log_event("Verifying downloaded firmware");
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        auto verification_result = device.staged_firmware->verify_integrity();
        if (!verification_result) {
            device.current_state = UpdateState::FAILED;
            device.log_event(std::format("Verification failed: {}", verification_result.error().message));
            return;
        }
        
        // Installation phase
        device.current_state = UpdateState::INSTALLING;
        device.log_event("Installing firmware");
        
        // Simulate power failure
        std::uniform_real_distribution<> power_failure_dis(0.0, 1.0);
        if (power_failure_dis(rng_) < 0.03) { // 3% chance of power failure
            device.current_state = UpdateState::RECOVERY_MODE;
            device.log_event("Power failure during installation - entering recovery mode");
            
            std::this_thread::sleep_for(std::chrono::seconds(2));
            auto rollback_result = device.rollback();
            if (rollback_result) {
                device.current_state = UpdateState::ROLLED_BACK;
            } else {
                device.current_state = UpdateState::FAILED;
                device.log_event("Recovery failed - device may require manual intervention");
            }
            return;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(
            static_cast<int>(1000 / device.update_speed_factor)));
        
        auto install_result = device.install_staged_firmware();
        if (!install_result) {
            device.current_state = UpdateState::FAILED;
            device.log_event(std::format("Installation failed: {}", install_result.error().message));
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
    
    /**
     * Download firmware using chunked approach with enhanced error handling
     * @intuition: Reliable firmware download with progress tracking and retry logic
     * @approach: Chunk-by-chunk download with comprehensive error recovery
     * @complexity: Time O(total_chunks * avg_retries), Space O(firmware_size)
     */
    [[nodiscard]] Result<void> download_firmware_chunked(Device& device, const FirmwarePackage& firmware) {
        device.log_event(std::format("Downloading firmware {} ({} bytes)", 
                                    firmware.version.to_string(), firmware.total_size));
        
        auto total_chunks = ChunkedDownloader::calculate_total_chunks(firmware.total_size);
        std::vector<uint8_t> downloaded_data;
        downloaded_data.reserve(firmware.total_size);
        
        for (size_t chunk_idx = 0; chunk_idx < total_chunks; ++chunk_idx) {
            auto result = downloader_.download_chunk(firmware, chunk_idx);
            
            if (!result.success) {
                return std::unexpected(result.error.value_or(
                    ErrorContext{UpdateError::NetworkTimeout, "Unknown download error"}));
            }
            
            downloaded_data.insert(downloaded_data.end(), 
                                 result.chunk_data.begin(), result.chunk_data.end());
            
            // Progress updates
            if (chunk_idx % 10 == 0 || chunk_idx == total_chunks - 1) {
                auto progress = ((chunk_idx + 1) * 100) / total_chunks;
                device.log_event(std::format("Download progress: {}%", progress));
            }
        }
        
        // Create staged firmware
        auto staged = std::make_unique<FirmwarePackage>(firmware.version, firmware.total_size);
        staged->data = std::move(downloaded_data);
        staged->checksum = firmware.checksum;
        
        auto stage_result = device.stage_firmware(std::move(staged));
        if (!stage_result) {
            return std::unexpected(stage_result.error());
        }
        
        device.log_event("Firmware download completed successfully");
        return {};
    }
    
    /**
     * Convert device type to human-readable string
     * @intuition: Provide readable device type information for display
     * @approach: Static mapping with comprehensive type coverage
     * @complexity: Time O(1), Space O(1)
     */
    [[nodiscard]] static constexpr std::string_view device_type_to_string(DeviceType type) noexcept {
        switch (type) {
            case DeviceType::SENSOR_LOW_POWER: return "Sensor";
            case DeviceType::GATEWAY_HIGH_PERFORMANCE: return "Gateway";
            case DeviceType::ACTUATOR_REAL_TIME: return "Actuator";
        }
        return "Unknown";
    }
    
    /**
     * Convert update state to human-readable string
     * @intuition: Provide readable state information for monitoring and debugging
     * @approach: Comprehensive state mapping with all possible states covered
     * @complexity: Time O(1), Space O(1)
     */
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
            case UpdateState::CANCELLED: return "Cancelled";
        }
        return "Unknown";
    }
};

/**
 * Command-line interface for interactive OTA management with enhanced user experience
 * @intuition: Provide comprehensive real-time monitoring and control capabilities
 * @approach: Menu-driven CLI with robust input validation and error handling
 * @complexity: Time O(1) per command, Space O(command_history)
 */
class CLI {
private:
    OTAManager& ota_manager_;
    bool running_{true};
    
public:
    /**
     * Initialize CLI with OTA manager reference
     * @intuition: Provide interactive interface to OTA system
     * @approach: Reference-based coupling for direct system access
     * @complexity: Time O(1), Space O(1)
     */
    explicit CLI(OTAManager& manager) : ota_manager_(manager) {}
    
    /**
     * Main CLI event loop with comprehensive menu system
     * @intuition: Interactive user interface for system control and monitoring
     * @approach: Menu-driven interface with input validation and error handling
     * @complexity: Time O(∞) interactive, Space O(1)
     */
    void run() {
        std::cout << "=== Advanced OTA Update Simulator ===\n";
        std::cout << "Enterprise-Grade IoT Firmware Management System\n";
        std::cout << "Features: Chunked Downloads | Rollback Support | Real-time Monitoring\n\n";
        
        while (running_) {
            print_menu();
            process_command();
        }
    }
    
private:
    /**
     * Display interactive menu options
     * @intuition: Clear presentation of available system operations
     * @approach: Numbered menu with descriptive option labels
     * @complexity: Time O(1), Space O(1)
     */
    void print_menu() const {
        std::cout << "\n╔══════════════════════════════════════╗\n";
        std::cout << "║            Main Menu                 ║\n";
        std::cout << "╠══════════════════════════════════════╣\n";
        std::cout << "║ 1. List all devices                  ║\n";
        std::cout << "║ 2. Show device details               ║\n";
        std::cout << "║ 3. Queue firmware update             ║\n";
        std::cout << "║ 4. Rollback device firmware          ║\n";
        std::cout << "║ 5. Monitor updates (live view)       ║\n";
        std::cout << "║ 6. Add test devices & firmware       ║\n";
        std::cout << "║ 7. System statistics                 ║\n";
        std::cout << "║ 8. Exit application                  ║\n";
        std::cout << "╚══════════════════════════════════════╝\n";
        std::cout << "Enter your choice (1-8): ";
    }
    
    /**
     * Process user command with input validation
     * @intuition: Route user input to appropriate system operations
     * @approach: Switch-based command routing with error handling
     * @complexity: Time O(1) for routing, varies by operation, Space O(1)
     */
    void process_command() {
        int choice;
        if (!(std::cin >> choice)) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "❌ Invalid input. Please enter a number between 1-8.\n";
            return;
        }
        
        switch (choice) {
            case 1: ota_manager_.print_device_status(); break;
            case 2: show_device_details(); break;
            case 3: queue_update_interactive(); break;
            case 4: rollback_device_interactive(); break;
            case 5: monitor_updates(); break;
            case 6: add_test_devices(); break;
            case 7: show_system_statistics(); break;
            case 8: 
                running_ = false; 
                std::cout << "👋 Shutting down OTA Update Simulator...\n";
                break;
            default: 
                std::cout << "❌ Invalid choice. Please select 1-8.\n";
        }
    }
    
    /**
     * Display comprehensive device information
     * @intuition: Provide detailed device status for troubleshooting
     * @approach: Interactive device selection with formatted output
     * @complexity: Time O(log_entries), Space O(1)
     */
    void show_device_details() {
        std::cout << "Enter device ID: ";
        uint32_t device_id;
        if (!(std::cin >> device_id)) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "❌ Invalid device ID format\n";
            return;
        }
        
        auto devices = ota_manager_.list_devices();
        auto device_it = std::ranges::find_if(devices, 
            [device_id](const Device* d) { return d->device_id == device_id; });
        
        if (device_it == devices.end()) {
            std::cout << "❌ Device not found\n";
            return;
        }
        
        const auto& device = **device_it;
        std::cout << std::format("\n╔═══ Device {} Details ═══╗\n", device_id);
        std::cout << std::format("║ Name: {:<25} ║\n", device.name);
        std::cout << std::format("║ Type: {:<25} ║\n", device_type_to_string(device.type));
        std::cout << std::format("║ Current Version: {:<13} ║\n", device.current_version.to_string());
        std::cout << std::format("║ Previous Version: {:<12} ║\n", device.previous_version.to_string());
        std::cout << std::format("║ Current State: {:<15} ║\n", update_state_to_string(device.current_state.load()));
        std::cout << "╚═══════════════════════════════╝\n";
        
        std::cout << "\n📋 Recent Activity Log:\n";
        auto logs = device.get_recent_logs(8);
        for (const auto& log : logs) {
            std::cout << "   " << log << "\n";
        }
    }
    
    /**
     * Interactive firmware update queuing
     * @intuition: Guide user through update process with validation
     * @approach: Step-by-step input collection with immediate feedback
     * @complexity: Time O(1), Space O(1)
     */
    void queue_update_interactive() {
        std::cout << "📱 Enter device ID: ";
        uint32_t device_id;
        if (!(std::cin >> device_id)) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "❌ Invalid device ID format\n";
            return;
        }
        
        std::cout << "🔢 Enter target version (format: major.minor.patch): ";
        std::string version_str;
        std::cin >> version_str;
        
        auto version_result = FirmwareVersion::parse(version_str);
        if (!version_result) {
            std::cout << std::format("❌ {}\n", version_result.error().message);
            return;
        }
        
        auto queue_result = ota_manager_.queue_update(device_id, version_result.value());
        if (!queue_result) {
            std::cout << std::format("❌ Failed to queue update: {}\n", queue_result.error().message);
        } else {
            std::cout << "✅ Update queued successfully!\n";
        }
    }
    
    /**
     * Interactive device rollback operation
     * @intuition: Safe rollback with user confirmation and validation
     * @approach: Device selection with immediate rollback execution
     * @complexity: Time O(1), Space O(1)
     */
    void rollback_device_interactive() {
        std::cout << "🔄 Enter device ID to rollback: ";
        uint32_t device_id;
        if (!(std::cin >> device_id)) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "❌ Invalid device ID format\n";
            return;
        }
        
        auto rollback_result = ota_manager_.rollback_device(device_id);
        if (!rollback_result) {
            std::cout << std::format("❌ Rollback failed: {}\n", rollback_result.error().message);
        } else {
            std::cout << "✅ Device rolled back successfully!\n";
        }
    }
    
    /**
     * Real-time update monitoring with live refresh
     * @intuition: Continuous system monitoring for operations oversight
     * @approach: Periodic screen refresh with live status updates
     * @complexity: Time O(monitoring_duration), Space O(1)
     */
    void monitor_updates() {
        std::cout << "📊 Starting live update monitor (30 second timeout)...\n";
        std::cout << "Press Ctrl+C to interrupt monitoring\n\n";
        
        auto start_time = std::chrono::steady_clock::now();
        int refresh_count = 0;
        
        while (true) {
            // Clear screen
            std::cout << "\033[2J\033[H";
            
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time);
            
            std::cout << std::format("🔴 LIVE MONITOR - Refresh #{} ({}s elapsed)\n", 
                                   ++refresh_count, elapsed.count());
            std::cout << std::string(60, '=') << "\n";
            
            ota_manager_.print_device_status();
            
            if (elapsed.count() >= 30) {
                std::cout << "\n⏰ Monitor timeout reached.\n";
                std::cout << "Press Enter to return to main menu...";
                std::cin.ignore();
                std::cin.get();
                break;
            }
            
            std::this_thread::sleep_for(std::chrono::seconds(3));
        }
    }
    
    /**
     * Add comprehensive test data for demonstration
     * @intuition: Populate system with realistic test scenarios
     * @approach: Create diverse device types and firmware versions
     * @complexity: Time O(1), Space O(test_data_size)
     */
    void add_test_devices() {
        std::cout << "🧪 Adding test devices and firmware packages...\n";
        
        // Create diverse test devices
        auto sensor1 = std::make_unique<Device>(DeviceType::SENSOR_LOW_POWER, 
                                               "Environmental Sensor Hub", 
                                               FirmwareVersion{1, 2, 1});
        
        auto sensor2 = std::make_unique<Device>(DeviceType::SENSOR_LOW_POWER,
                                               "Motion Detection Unit",
                                               FirmwareVersion{1, 0, 5});
        
        auto gateway1 = std::make_unique<Device>(DeviceType::GATEWAY_HIGH_PERFORMANCE,
                                                "Primary IoT Gateway",
                                                FirmwareVersion{2, 1, 0});
        
        auto gateway2 = std::make_unique<Device>(DeviceType::GATEWAY_HIGH_PERFORMANCE,
                                                "Secondary Gateway Node",
                                                FirmwareVersion{2, 0, 8});
        
        auto actuator1 = std::make_unique<Device>(DeviceType::ACTUATOR_REAL_TIME,
                                                 "Smart Valve Controller",
                                                 FirmwareVersion{1, 5, 2});
        
        auto actuator2 = std::make_unique<Device>(DeviceType::ACTUATOR_REAL_TIME,
                                                 "HVAC Control Module",
                                                 FirmwareVersion{1, 4, 9});
        
        // Add devices to manager
        ota_manager_.add_device(std::move(sensor1));
        ota_manager_.add_device(std::move(sensor2));
        ota_manager_.add_device(std::move(gateway1));
        ota_manager_.add_device(std::move(gateway2));
        ota_manager_.add_device(std::move(actuator1));
        ota_manager_.add_device(std::move(actuator2));
        
        // Add diverse firmware versions
        auto fw_1_3_0 = std::make_unique<FirmwarePackage>(FirmwareVersion{1, 3, 0}, 45056);   // Sensor update
        auto fw_1_6_0 = std::make_unique<FirmwarePackage>(FirmwareVersion{1, 6, 0}, 52224);   // Actuator update
        auto fw_2_2_0 = std::make_unique<FirmwarePackage>(FirmwareVersion{2, 2, 0}, 131072);  // Gateway update
        auto fw_2_3_0 = std::make_unique<FirmwarePackage>(FirmwareVersion{2, 3, 0}, 147456);  // Latest gateway
        auto fw_3_0_0 = std::make_unique<FirmwarePackage>(FirmwareVersion{3, 0, 0}, 98304);   // Major version
        
        ota_manager_.add_firmware(std::move(fw_1_3_0));
        ota_manager_.add_firmware(std::move(fw_1_6_0));
        ota_manager_.add_firmware(std::move(fw_2_2_0));
        ota_manager_.add_firmware(std::move(fw_2_3_0));
        ota_manager_.add_firmware(std::move(fw_3_0_0));
        
        std::cout << "✅ Test environment setup complete!\n";
        std::cout << "   • 6 devices added (2 sensors, 2 gateways, 2 actuators)\n";
        std::cout << "   • 5 firmware versions added\n";
        std::cout << "   • Ready for update testing\n";
    }
    
    /**
     * Display comprehensive system statistics
     * @intuition: Provide system health and performance metrics
     * @approach: Aggregate device states and compute statistics
     * @complexity: Time O(n) where n is device count, Space O(1)
     */
    void show_system_statistics() {
        auto devices = ota_manager_.list_devices();
        
        if (devices.empty()) {
            std::cout << "📊 No devices in system\n";
            return;
        }
        
        // Count devices by type and state
        std::unordered_map<DeviceType, size_t> type_counts;
        std::unordered_map<UpdateState, size_t> state_counts;
        
        for (const auto* device : devices) {
            type_counts[device->type]++;
            state_counts[device->current_state.load()]++;
        }
        
        std::cout << "\n📊 System Statistics\n";
        std::cout << std::string(40, '=') << "\n";
        std::cout << std::format("Total Devices: {}\n", devices.size());
        
        std::cout << "\nDevice Types:\n";
        for (const auto& [type, count] : type_counts) {
            std::cout << std::format("  {}: {} devices\n", 
                                   device_type_to_string(type), count);
        }
        
        std::cout << "\nCurrent States:\n";
        for (const auto& [state, count] : state_counts) {
            std::cout << std::format("  {}: {} devices\n", 
                                   update_state_to_string(state), count);
        }
        
        // Calculate success rate
        auto success_count = state_counts[UpdateState::SUCCESS];
        auto failed_count = state_counts[UpdateState::FAILED];
        auto total_completed = success_count + failed_count;
        
        if (total_completed > 0) {
            auto success_rate = (success_count * 100.0) / total_completed;
            std::cout << std::format("\nUpdate Success Rate: {:.1f}% ({}/{} completed)\n", 
                                   success_rate, success_count, total_completed);
        }
    }
    
    /**
     * Convert device type enum to descriptive string
     * @intuition: Human-readable device type labels for user interface
     * @approach: Comprehensive type mapping with detailed descriptions
     * @complexity: Time O(1), Space O(1)
     */
    [[nodiscard]] static constexpr std::string_view device_type_to_string(DeviceType type) noexcept {
        switch (type) {
            case DeviceType::SENSOR_LOW_POWER: return "Low Power Sensor";
            case DeviceType::GATEWAY_HIGH_PERFORMANCE: return "High Performance Gateway";
            case DeviceType::ACTUATOR_REAL_TIME: return "Real-Time Actuator";
        }
        return "Unknown Device Type";
    }
    
    /**
     * Convert update state enum to descriptive string
     * @intuition: Clear state representation for user monitoring
     * @approach: Complete state mapping with user-friendly labels
     * @complexity: Time O(1), Space O(1)
     */
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
            case UpdateState::CANCELLED: return "Cancelled";
        }
        return "Unknown State";
    }
};

} // namespace ota_simulator

/**
 * Application entry point with comprehensive OTA simulation setup
 * @intuition: Initialize complete system with professional user experience
 * @approach: Exception-safe initialization with graceful error handling and user guidance
 * @complexity: Time O(1) initialization, Space O(system_components)
 */
int main() {
    using namespace ota_simulator;
    
    try {
        std::cout << "🚀 Initializing Advanced OTA Update Simulator...\n";
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        
        OTAManager ota_manager;
        CLI cli(ota_manager);
        
        std::cout << "✅ System Features:\n";
        std::cout << "   • Multi-device type support with performance scaling\n";
        std::cout << "   • Chunked downloads with bandwidth optimization\n";
        std::cout << "   • Network interruption and power failure simulation\n";
        std::cout << "   • Automatic rollback mechanisms with integrity checking\n";
        std::cout << "   • Thread-safe operations with granular locking\n";
        std::cout << "   • Comprehensive logging and real-time monitoring\n";
        std::cout << "   • Type-safe error handling with detailed context\n";
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        
        cli.run();
        
        std::cout << "\n🏁 OTA Update Simulator shutdown complete.\n";
        std::cout << "Thank you for using the Enterprise IoT Firmware Management System!\n";
        
    } catch (const std::exception& e) {
        std::cerr << std::format("💥 Fatal system error: {}\n", e.what());
        std::cerr << "Please check system requirements and try again.\n";
        return 1;
    } catch (...) {
        std::cerr << "💥 Unknown fatal error occurred\n";
        std::cerr << "System state may be corrupted. Please restart.\n";
        return 2;
    }
    
    return 0;
}
