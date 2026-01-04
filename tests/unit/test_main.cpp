#include "absl/log/initialize.h"
#include "gtest/gtest.h"

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  absl::InitializeLog();
  return RUN_ALL_TESTS();
}
