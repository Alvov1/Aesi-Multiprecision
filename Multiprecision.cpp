#include "Multiprecision.h"

std::ostream &operator<<(std::ostream &stream, const Multiprecision<> &value) noexcept {
    /*if (value.getSign() == Negative)
        stream << '-';

    int print_zeroes = 0;
    for (int i = value.blocks.size() - 1; i >= 0; i--) {
        unsigned digit = value.blocks[i];

        static constexpr auto hexDigits = "0123456789abcdef";
        if (digit != 0 || print_zeroes) {
            if (!print_zeroes) {
                char buffer[9] = {'0', '0', '0', '0', '0', '0', '0', '0', '\0'};
                uint8_t bufferPosition = 0;

                while(digit > 0 && bufferPosition < 8) {
                    buffer[bufferPosition++] = hexDigits[digit % 16];
                    digit /= 16;
                }

                for(uint8_t j = 0; j < bufferPosition; ++j)
                    stream << buffer[bufferPosition - j - 1];
            } else {
                char buffer[9] {};
                uint8_t bufferPosition = 0;

                while(digit > 0 && bufferPosition < 8) {
                    buffer[bufferPosition++] = hexDigits[digit % 16];
                    digit /= 16;
                }

                for(uint8_t j = 0; j < 8; ++j)
                    stream << buffer[7 - j];
            }
            print_zeroes = 1;
        }
    }

    if (print_zeroes == 0)
        stream << '0';*/
    return stream;
}
