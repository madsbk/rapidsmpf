/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @brief GTest for rapidsmpf::apply_hardware_bindings API.
 *
 * These tests must be run with rrun so that CUDA_VISIBLE_DEVICES is set and a
 * specific GPU is assigned to the process. Example:
 *   rrun -n 1 gtests/hardware_binding_tests
 */

#include <algorithm>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cucascade/memory/topology_discovery.hpp>

#include <rapidsmpf/bootstrap/utils.hpp>
#include <rapidsmpf/utils/hardware_binding.hpp>

class HardwareBindingsTest : public ::testing::Test {
  protected:
    void SetUp() override {
        if (!rapidsmpf::bootstrap::is_running_with_rrun()) {
            GTEST_SKIP() << "Test must be run with rrun (RAPIDSMPF_RANK not set)";
        }

        gpu_id_ = rapidsmpf::bootstrap::get_gpu_id();
        if (gpu_id_ < 0) {
            GTEST_SKIP() << "Could not determine GPU ID from CUDA_VISIBLE_DEVICES";
        }

        if (!discovery_.discover()) {
            GTEST_SKIP() << "Failed to discover topology";
        }

        auto const& topology = discovery_.get_topology();
        auto const it = std::ranges::find_if(
            topology.gpus, [this](cucascade::memory::gpu_topology_info const& gpu) {
                return static_cast<int>(gpu.id) == gpu_id_;
            }
        );

        if (it == topology.gpus.end()) {
            GTEST_SKIP() << "GPU ID " << gpu_id_ << " not found in topology";
        }

        gpu_info_ = *it;
    }

    cucascade::memory::topology_discovery discovery_;
    int gpu_id_{-1};
    cucascade::memory::gpu_topology_info gpu_info_;
};

TEST_F(HardwareBindingsTest, ApplyByIndex) {
    EXPECT_NO_THROW(rapidsmpf::apply_hardware_bindings(gpu_id_));
}

TEST_F(HardwareBindingsTest, ApplyByUuid) {
    if (gpu_info_.uuid.empty()) {
        GTEST_SKIP() << "GPU UUID not available for GPU " << gpu_id_;
    }
    EXPECT_NO_THROW(rapidsmpf::apply_hardware_bindings(gpu_info_.uuid));
}

TEST_F(HardwareBindingsTest, CpuOnlyOption) {
    rapidsmpf::HardwareBindingOptions opts;
    opts.cpu = true;
    opts.memory = false;
    opts.network = false;
    EXPECT_NO_THROW(rapidsmpf::apply_hardware_bindings(gpu_id_, opts));
}

TEST_F(HardwareBindingsTest, NetworkOnlyOption) {
    rapidsmpf::HardwareBindingOptions opts;
    opts.cpu = false;
    opts.memory = false;
    opts.network = true;
    EXPECT_NO_THROW(rapidsmpf::apply_hardware_bindings(gpu_id_, opts));
}

TEST(HardwareBindingsErrorTest, InvalidIndex) {
    EXPECT_THROW(rapidsmpf::apply_hardware_bindings(-1), std::logic_error);
    EXPECT_THROW(rapidsmpf::apply_hardware_bindings(99999), std::logic_error);
}

TEST(HardwareBindingsErrorTest, EmptyUuid) {
    EXPECT_THROW(rapidsmpf::apply_hardware_bindings(std::string("")), std::logic_error);
}
