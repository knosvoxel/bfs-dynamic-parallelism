#pragma once

#include <chrono>
#include <sstream>
#include <iomanip>

using Clock = std::chrono::steady_clock;
using TimePoint = std::chrono::time_point<Clock>;

class Timer
{
public:
    Timer() { Reset(); }

    inline void Reset() { m_start = Clock::now(); }
    inline TimePoint GetStartTime() const { return m_start; }

    inline double ElapsedMs() const { return std::chrono::duration<double, std::milli>(Clock::now() - m_start).count(); }
    inline double ElapsedSec() const { return std::chrono::duration<double>(Clock::now() - m_start).count(); }

    inline std::string ToString(const std::string& label) const {
        std::ostringstream oss;
        oss << "[Timer] " << label << ": "
            << std::fixed << std::setprecision(3)
            << std::setw(10) << ElapsedMs() << " ms";
        return oss.str();
    }

private:
    TimePoint m_start;
};