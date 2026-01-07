#pragma once

#include <cstddef>

// Simple cross-platform progress bar
void printProgressBar(std::size_t current, std::size_t total, int barWidth = 50);
