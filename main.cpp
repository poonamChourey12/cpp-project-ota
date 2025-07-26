struct FirmwareVersion {
    uint32_t major{0};
    uint32_t minor{0};
    uint32_t patch{0};
    constexpr auto operator<=>(const FirmwareVersion& other) const = default;
    // Do NOT declare operator== explicitly. operator<=> covers it.
    // constexpr bool operator==(const FirmwareVersion&) const = default; // REMOVE
    ...
};

class Device {
public:
    // ALL PUBLIC MEMBERS FIRST
    const uint32_t device_id = next_id_++;
    // ...other public members...
private:
    // ALL PRIVATE MEMBERS LAST
    static inline std::atomic<uint32_t> next_id_{1000};
    ...
};

// In CLI or similar class
void add_test_devices() const {  // Make const if not modifying members
    // ...
}

// C++23 escape (in monitor_updates)
std::cout << "\x1b[2J\x1b[H"; // instead of "\033[2J\033[H"
