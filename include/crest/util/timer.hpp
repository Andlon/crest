#pragma once

#include <chrono>

namespace crest
{
    /**
     * A simple timer that returns elapsed time in seconds since its construction.
     */
    class Timer
    {
    public:
        Timer()
        {
            _start = std::chrono::steady_clock::now();
        }

        double elapsed() const
        {
            const auto now = std::chrono::steady_clock::now();
            const std::chrono::duration<double> seconds = now - _start;
            return seconds.count();
        }

        void reset()
        {
            _start = std::chrono::steady_clock::now();
        }

        double measure_and_reset()
        {
            const auto t = elapsed();
            reset();
            return t;
        }

    private:
        std::chrono::steady_clock::time_point _start;
    };

    namespace detail
    {
        /**
         * Helper utility class that adds the elapsed time since construction to a variable upon destruction.
         * Helpful when writing generic code.
         */
        class RAIITimer
        {
        public:
            explicit RAIITimer(double & ref) : _ref(ref)
            {}

            ~RAIITimer() { _ref += _timer.elapsed(); }

        private:
            double &        _ref;
            ::crest::Timer  _timer;
        };
    }

    /**
     * Helper function to measure the time for a function call. The result is *added* to dest, meaning
     * that it must be initialized to zero for one-off measurements.
     * @param dest
     * @param f
     * @return
     */
    template <typename Func>
    auto inspect_timing(double & dest, Func f)
    {
        detail::RAIITimer timer(dest);
        return f();
    };
}
