#include <optional>
#include <stdexcept>

#include <gtest/gtest.h>

#include <Engine/Interface/GeometryBuffers.h>

namespace
{
    class TestGeometryBuffers : public _GeometryBuffers
    {
    public:
        bool isMemoryShareable() const override { return false; }
        SharedGeometryMemory shareMemory(GeometryBufferType type) override { return {}; }

        void upload(GeometryBufferType type, void const* data, uint64_t sizeInBytes) override {}
        void download(GeometryBufferType type, void* data, uint64_t sizeInBytes) const override {}

        std::optional<GeometryBufferType> failingType;

    protected:
        void reallocate(GeometryBufferType type, uint64_t sizeInBytes) override
        {
            if (failingType == type) {
                throw std::runtime_error("Out of memory");
            }
        }
    };
}

class GeometryBuffersTests : public ::testing::Test
{
public:
    GeometryBuffersTests() = default;
    virtual ~GeometryBuffersTests() = default;

protected:
    TestGeometryBuffers _geometryBuffers;
};

TEST_F(GeometryBuffersTests, growth_changesAllocationId)
{
    _geometryBuffers.updateNumObjects({});
    auto capacity = _geometryBuffers.getCapacity(GeometryBufferType_Objects);
    auto allocationId = _geometryBuffers.getAllocationId(GeometryBufferType_Objects);

    _geometryBuffers.updateNumObjects({.objects = capacity});

    EXPECT_GT(_geometryBuffers.getCapacity(GeometryBufferType_Objects), capacity);
    EXPECT_NE(allocationId, _geometryBuffers.getAllocationId(GeometryBufferType_Objects));
    EXPECT_EQ(capacity, _geometryBuffers.getNumObjects().objects);
}

TEST_F(GeometryBuffersTests, failedGrowth_keepsBuffer)
{
    _geometryBuffers.updateNumObjects({});
    auto capacity = _geometryBuffers.getCapacity(GeometryBufferType_Objects);
    auto allocationId = _geometryBuffers.getAllocationId(GeometryBufferType_Objects);

    _geometryBuffers.failingType = GeometryBufferType_Objects;
    EXPECT_THROW(_geometryBuffers.updateNumObjects({.objects = capacity}), std::runtime_error);

    EXPECT_EQ(capacity, _geometryBuffers.getCapacity(GeometryBufferType_Objects));
    EXPECT_EQ(allocationId, _geometryBuffers.getAllocationId(GeometryBufferType_Objects));
    EXPECT_EQ(0u, _geometryBuffers.getNumObjects().objects);
}
