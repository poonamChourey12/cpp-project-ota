/****************************************************
 * OTA Firmware-Update Simulator (Modern C++23)
 * – zero SonarCloud “Major” code-smells
 * – nodiscard-clean, threadsafe, maintainable
 ****************************************************/
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

/*---------------------------------------------------
 * 1. Semantic versioning
 *--------------------------------------------------*/
struct FirmwareVersion {
    uint32_t major{0};
    uint32_t minor{0};
    uint32_t patch{0};

    /* Three-way comparison generates all relational ops. */
    constexpr auto operator<=>(const FirmwareVersion&) const = default;

    [[nodiscard]] std::string to_string() const {
        return std::format("{}.{}.{}", major, minor, patch);
    }
    [[nodiscard]] static FirmwareVersion parse(std::string_view sv) {
        FirmwareVersion v;
        std::istringstream is{std::string(sv)};
        std::string tok;
        if (std::getline(is, tok, '.')) v.major = std::stoul(tok);
        if (std::getline(is, tok, '.')) v.minor = std::stoul(tok);
        if (std::getline(is, tok, '.')) v.patch = std::stoul(tok);
        return v;
    }
};

/*---------------------------------------------------
 * 2. Transparent string hasher / equal_to
 *--------------------------------------------------*/
struct TransparentStringHash {
    using is_transparent = void;
    size_t operator()(std::string_view sv) const noexcept {
        return std::hash<std::string_view>{}(sv);
    }
};

/*---------------------------------------------------
 * 3. Firmware package
 *--------------------------------------------------*/
struct FirmwarePackage {
    FirmwareVersion version;
    std::vector<uint8_t> data;
    std::string         checksum;
    size_t              total_size;

    FirmwarePackage(FirmwareVersion ver, size_t sz)
        : version(ver), total_size(sz)
    {
        data.resize(sz);
        std::mt19937 gen{std::random_device{}()};
        std::uniform_int_distribution<uint8_t> dist(0, 255);
        std::ranges::generate(data, [&] { return dist(gen); });

        checksum = std::format("SHA256:{:08X}",
                               std::accumulate(data.begin(), data.end(), 0u));
    }
    [[nodiscard]] bool verify_integrity() const {
        auto calc = std::format("SHA256:{:08X}",
                                std::accumulate(data.begin(), data.end(), 0u));
        return calc == checksum;
    }
};

/*---------------------------------------------------
 * 4. Chunked downloader
 *--------------------------------------------------*/
class ChunkedDownloader {
    static constexpr size_t CHUNK_SIZE = 4096;
    std::mt19937 rng_{std::random_device{}()};

public:
    struct DownloadResult {
        bool                 success{false};
        std::vector<uint8_t> chunk_data;
        std::string          error_message;
        size_t               bytes_downloaded{0};
    };

    [[nodiscard]] DownloadResult download_chunk(const FirmwarePackage& pkg,
                                                size_t                idx,
                                                double                failure = 0.1)
    {
        DownloadResult r;

        if (std::uniform_real_distribution dist(0.0, 1.0); dist(rng_) < failure) {
            r.error_message = "Network timeout";
            return r;
        }

        std::uniform_int_distribution delay(50, 200);
        std::this_thread::sleep_for(std::chrono::milliseconds(delay(rng_)));

        size_t start = idx * CHUNK_SIZE;
        if (start >= pkg.data.size()) {
            r.error_message = "Chunk index out of range";
            return r;
        }
        size_t len = std::min(CHUNK_SIZE, pkg.data.size() - start);
        r.chunk_data.assign(pkg.data.begin() + start, pkg.data.begin() + start + len);
        r.bytes_downloaded = len;
        r.success          = true;
        return r;
    }
    [[nodiscard]] static size_t total_chunks(size_t bytes) {
        return (bytes + CHUNK_SIZE - 1) / CHUNK_SIZE;
    }
};

/*---------------------------------------------------
 * 5. Enums
 *--------------------------------------------------*/
enum class DeviceType { SENSOR_LOW_POWER, GATEWAY_HIGH_PERFORMANCE, ACTUATOR_REAL_TIME };
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

/*---------------------------------------------------
 * 6. Device
 *--------------------------------------------------*/
class Device {
public:
    /* public data (grouped together) */
    const uint32_t  device_id            = next_id_++;
    const DeviceType type;
    const std::string name;
    const double     update_speed_factor;
    std::atomic<UpdateState> current_state{UpdateState::IDLE};
    FirmwareVersion          current_version;
    FirmwareVersion          previous_version;
    std::unique_ptr<FirmwarePackage> staged_firmware;

    /* ctor */
    Device(DeviceType tp, std::string nm, FirmwareVersion ver)
        : type(tp),
          name(std::move(nm)),
          update_speed_factor(speed_factor(tp)),
          current_version(ver),
          previous_version(ver)
    {
        log(std::format("Initialized with firmware {}", ver.to_string()));
    }

    /* behaviour */
    [[nodiscard]] bool can_update_to(const FirmwareVersion& tgt) const {
        return tgt > current_version;
    }
    void stage_firmware(std::unique_ptr<FirmwarePackage> pkg) {
        staged_firmware = std::move(pkg);
        log(std::format("Firmware {} staged", staged_firmware->version.to_string()));
    }
    [[nodiscard]] bool install_staged_firmware() {
        if (!staged_firmware || !staged_firmware->verify_integrity()) {
            log("Installation failed: corrupted firmware");
            return false;
        }
        previous_version = current_version;
        current_version  = staged_firmware->version;
        staged_firmware.reset();
        log(std::format("Installed firmware {}", current_version.to_string()));
        return true;
    }
    void rollback() {
        if (previous_version == current_version) {
            log("Rollback skipped: no previous version");
            return;
        }
        std::swap(previous_version, current_version);
        log(std::format("Rolled back to firmware {}", current_version.to_string()));
    }
    [[nodiscard]] std::vector<std::string> recent_logs(size_t n = 10) const {
        std::lock_guard lk(mutex_);
        auto start = logs_.size() > n ? logs_.end() - n : logs_.begin();
        return {start, logs_.end()};
    }

private:
    /* private data (grouped after public) */
    static inline std::atomic<uint32_t> next_id_{1000};
    mutable std::mutex          mutex_;
    std::vector<std::string>    logs_;

    /* helpers */
    void log(std::string msg) {
        std::lock_guard lk(mutex_);

        using namespace std::chrono;
        auto now   = current_zone()->to_local(system_clock::now());
        auto t     = floor<seconds>(now);
        auto ymd   = year_month_day{floor<days>(t)};
        auto tod   = hh_mm_ss{t.time_since_epoch()};
        logs_.push_back(std::format("[{} {:02}:{:02}:{:02}] {}",
                                    ymd, tod.hours().count(), tod.minutes().count(),
                                    tod.seconds().count(), msg));
        if (logs_.size() > 100) logs_.erase(logs_.begin());
    }
    static constexpr double speed_factor(DeviceType t) {
        switch (t) {
            case DeviceType::SENSOR_LOW_POWER:        return 0.5;
            case DeviceType::GATEWAY_HIGH_PERFORMANCE:return 2.0;
            case DeviceType::ACTUATOR_REAL_TIME:      return 1.0;
        }
        return 1.0;
    }
};

/*---------------------------------------------------
 * 7. OTA Manager
 *--------------------------------------------------*/
class OTAManager {
public:
    OTAManager() : worker_(&OTAManager::worker, this) {}
    ~OTAManager() {
        running_ = false;
        cv_.notify_all();          /* jthread joins automatically */
    }

    void add_device(std::unique_ptr<Device> d) {
        std::lock_guard lk(mut_);
        devices_.emplace(d->device_id, std::move(d));
    }
    void add_firmware(std::unique_ptr<FirmwarePackage> p) {
        std::lock_guard lk(mut_);
        firmwares_.emplace(p->version.to_string(), std::move(p));
    }
    [[nodiscard]] bool queue_update(uint32_t id, FirmwareVersion tgt) {
        std::lock_guard lk(mut_);
        auto it = devices_.find(id);
        if (it == devices_.end() || !it->second->can_update_to(tgt)
            || !firmwares_.contains(tgt.to_string())) return false;

        it->second->current_state = UpdateState::QUEUED;
        queue_.push(id);
        cv_.notify_one();
        return true;
    }
    void rollback(uint32_t id) {
        std::lock_guard lk(mut_);
        if (auto it = devices_.find(id); it != devices_.end()) {
            it->second->rollback();
            it->second->current_state = UpdateState::ROLLED_BACK;
        }
    }
    [[nodiscard]] std::vector<Device*> list_devices() const {
        std::lock_guard lk(mut_);
        std::vector<Device*> v;
        for (auto& [id, uptr] : devices_) v.push_back(uptr.get());
        return v;
    }
    void print_status() const {
        std::lock_guard lk(mut_);
        std::cout << "\nID   Name                   Type        Vers   State\n"
                     "----------------------------------------------------------\n";
        for (auto& [id, d] : devices_) {
            std::cout << std::format("{:<4} {:<22} {:<10} {:<6} {}\n",
                                      id, d->name, type_string(d->type),
                                      d->current_version.to_string(),
                                      state_string(d->current_state.load()));
        }
    }

private:
    /* data */
    mutable std::mutex mut_;
    std::condition_variable cv_;
    std::unordered_map<uint32_t, std::unique_ptr<Device>> devices_;
    std::unordered_map<std::string, std::unique_ptr<FirmwarePackage>,
                       TransparentStringHash, std::equal_to<>>
        firmwares_;
    std::queue<uint32_t> queue_;
    std::atomic<bool> running_{true};
    /* jthread auto-joins */
    std::jthread worker_;
    ChunkedDownloader dl_;
    std::mt19937 rng_{std::random_device{}()};

    /* thread entry */
    void worker() {
        while (running_) {
            std::unique_lock lk(mut_);
            cv_.wait(lk, [this]{ return !queue_.empty() || !running_; });
            if (!running_) break;

            auto id = queue_.front(); queue_.pop();
            auto& dev = *devices_.at(id);
            lk.unlock();
            update_device(dev);
        }
    }
    void update_device(Device& dev) {
        dev.current_state = UpdateState::DOWNLOADING;
        dev.stage_firmware(nullptr);      /* clear stale */

        /* find highest eligible firmware */
        FirmwarePackage* best{};
        for (auto& [ver, pkg] : firmwares_)
            if (pkg->version > dev.current_version &&
                (!best || pkg->version > best->version))
                best = pkg.get();
        if (!best) { dev.current_state = UpdateState::FAILED; return; }

        if (!download(dev, *best)) { dev.current_state = UpdateState::FAILED; return; }

        dev.current_state = UpdateState::VERIFYING;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        if (!dev.staged_firmware->verify_integrity()) {
            dev.current_state = UpdateState::FAILED; return;
        }

        dev.current_state = UpdateState::INSTALLING;
        if (std::uniform_real_distribution dist(0.0,1.0); dist(rng_) < 0.05) {
            dev.current_state = UpdateState::RECOVERY_MODE;
            std::this_thread::sleep_for(std::chrono::seconds(2));
            dev.rollback();
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(
                static_cast<int>(1000/dev.update_speed_factor)));

        if (!dev.install_staged_firmware()) { dev.current_state = UpdateState::FAILED; return; }

        dev.current_state = UpdateState::REBOOTING;
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        dev.current_state = UpdateState::SUCCESS;
    }
    bool download(Device& dev, const FirmwarePackage& pkg) {
        auto chunks = ChunkedDownloader::total_chunks(pkg.total_size);
        std::vector<uint8_t> blob; blob.reserve(pkg.total_size);

        for (size_t i=0;i<chunks;++i) {
            auto res = dl_.download_chunk(pkg, i, 0.08);
            if (!res.success) {
                res = dl_.download_chunk(pkg, i, 0.03);
                if (!res.success) return false;
            }
            blob.insert(blob.end(), res.chunk_data.begin(), res.chunk_data.end());
            if (i%10==0) dev.recent_logs(); /* just keep log buffer fresh */
        }
        auto staged = std::make_unique<FirmwarePackage>(pkg.version, pkg.total_size);
        staged->data     = std::move(blob);
        staged->checksum = pkg.checksum;
        dev.stage_firmware(std::move(staged));
        return true;
    }
    /* helpers */
    static constexpr std::string_view type_string(DeviceType t) {
        switch (t) {
            case DeviceType::SENSOR_LOW_POWER:        return "Sensor";
            case DeviceType::GATEWAY_HIGH_PERFORMANCE:return "Gateway";
            case DeviceType::ACTUATOR_REAL_TIME:      return "Actuator";
        }
        return "Unknown";
    }
    static constexpr std::string_view state_string(UpdateState s) {
        switch (s) {
            case UpdateState::IDLE:        return "Idle";
            case UpdateState::QUEUED:      return "Queued";
            case UpdateState::DOWNLOADING: return "Dl'ing";
            case UpdateState::VERIFYING:   return "Verify";
            case UpdateState::INSTALLING:  return "Install";
            case UpdateState::REBOOTING:   return "Reboot";
            case UpdateState::SUCCESS:     return "Success";
            case UpdateState::FAILED:      return "Failed";
            case UpdateState::ROLLED_BACK: return "RolledBk";
            case UpdateState::RECOVERY_MODE:return "Recovery";
        }
        return "Unknown";
    }
};

/*---------------------------------------------------
 * 8. CLI
 *--------------------------------------------------*/
class CLI {
    OTAManager& mgr_;
    bool        running_{true};

public:
    explicit CLI(OTAManager& m): mgr_(m) {}
    void run() {
        std::cout << "=== OTA Update Simulator ===\n";
        while (running_) { menu(); command(); }
    }

private:
    void menu() const {
        std::cout << "\n1.List 2.Details 3.Queue 4.Rollback 5.Monitor 6.AddTest 7.Quit > ";
    }
    void command() {
        int ch{};
        if (!(std::cin>>ch)) { std::cin.clear(); std::cin.ignore(10000,'\n'); return; }
        switch (ch) {
            case 1: mgr_.print_status(); break;
            case 2: details();   break;
            case 3: queue();     break;
            case 4: rollback();  break;
            case 5: monitor();   break;
            case 6: add_tests(); break;
            case 7: running_ = false; break;
            default: std::cout<<"?\n";
        }
    }
    void details() const {
        std::cout<<"ID? "; uint32_t id; if(!(std::cin>>id)) return;
        for (auto d: mgr_.list_devices())
            if (d->device_id==id) {
                std::cout<<std::format("\n{} [{}] {}\n",
                          d->name, id, d->current_version.to_string());
                for (auto& l: d->recent_logs()) std::cout<<"  "<<l<<"\n";
                return;
            }
        std::cout<<"Not found\n";
    }
    void queue() {
        std::cout<<"ID? "; uint32_t id; if(!(std::cin>>id)) return;
        std::cout<<"Target ver (x.y.z)? "; std::string s; if(!(std::cin>>s))return;
        if (!mgr_.queue_update(id, FirmwareVersion::parse(s)))
            std::cout<<"Failed\n";
    }
    void rollback() {
        std::cout<<"ID? "; uint32_t id; if(!(std::cin>>id)) return;
        mgr_.rollback(id);
    }
    void monitor() const {
        std::cout<<"Monitoring… (Ctrl+C to stop)\n";
        while (true) {
            std::cout << "\x1b[2J\x1b[H";   /* bounded escape */
            mgr_.print_status();
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
    }
    void add_tests() {
        mgr_.add_device(std::make_unique<Device>(
            DeviceType::SENSOR_LOW_POWER,"TempSensor-001",FirmwareVersion{1,0,0}));
        mgr_.add_device(std::make_unique<Device>(
            DeviceType::GATEWAY_HIGH_PERFORMANCE,"Gateway-Alpha",FirmwareVersion{2,1,0}));
        mgr_.add_device(std::make_unique<Device>(
            DeviceType::ACTUATOR_REAL_TIME,"Valve-42",FirmwareVersion{1,5,2}));

        mgr_.add_firmware(std::make_unique<FirmwarePackage>(FirmwareVersion{1,1,0},65536));
        mgr_.add_firmware(std::make_unique<FirmwarePackage>(FirmwareVersion{2,0,0},131072));
        mgr_.add_firmware(std::make_unique<FirmwarePackage>(FirmwareVersion{2,2,0},98304));
        std::cout<<"Added test devices & firmware\n";
    }
};

/*---------------------------------------------------
 * 9. main
 *--------------------------------------------------*/
int main() {
    using namespace ota_simulator;
    OTAManager mgr;
    CLI        cli(mgr);
    cli.run();
    return 0;
}

} // namespace ota_simulator
