#include <cmath>

#include <gtest/gtest.h>

#include <Video/SmoothCurve.h>

class SmoothCurveTests : public ::testing::Test
{
protected:
    std::vector<SmoothCurve::Keyframe> createZigzag() const
    {
        return {
            {.time = 0, .values = {0, 1}},
            {.time = 1, .values = {10, 2}},
            {.time = 3, .values = {-5, 4}},
            {.time = 4, .values = {0, 8}},
        };
    }

    // Convolution of the unsmoothed curve with the Gaussian kernel by numerical integration
    double integrateSmoothedValue(SmoothCurve const& unsmoothedCurve, int channel, double time, double smoothing) const
    {
        auto constexpr NumSteps = 20000;
        auto range = 8 * smoothing;
        auto stepSize = 2 * range / NumSteps;
        auto weightedSum = 0.0;
        auto weightSum = 0.0;
        for (int step = 0; step <= NumSteps; ++step) {
            auto offset = -range + step * stepSize;
            auto weight = std::exp(-0.5 * offset * offset / (smoothing * smoothing));
            weightedSum += weight * unsmoothedCurve.getValues(time + offset).at(channel);
            weightSum += weight;
        }
        return weightedSum / weightSum;
    }
};

TEST_F(SmoothCurveTests, withoutSmoothing_interpolatesLinearly)
{
    SmoothCurve curve(createZigzag(), 0);

    EXPECT_DOUBLE_EQ(0, curve.getValues(0).at(0));
    EXPECT_DOUBLE_EQ(10, curve.getValues(1).at(0));
    EXPECT_DOUBLE_EQ(2.5, curve.getValues(2).at(0));
    EXPECT_DOUBLE_EQ(6, curve.getValues(3.5).at(1));
    EXPECT_DOUBLE_EQ(8, curve.getValues(4).at(1));
}

TEST_F(SmoothCurveTests, withoutSmoothing_continuesPointSymmetrically)
{
    SmoothCurve curve(createZigzag(), 0);

    EXPECT_NEAR(-10, curve.getValues(-1).at(0), 1e-9);
    EXPECT_NEAR(0, curve.getValues(-1).at(1), 1e-9);
    EXPECT_NEAR(5, curve.getValues(5).at(0), 1e-9);
    EXPECT_NEAR(12, curve.getValues(5).at(1), 1e-9);

    // Mirrored at the start, the continuation beyond the end follows
    EXPECT_NEAR(-5, curve.getValues(-5).at(0), 1e-9);
    EXPECT_NEAR(-10, curve.getValues(-5).at(1), 1e-9);
}

TEST_F(SmoothCurveTests, smoothing_keepsStartAndEnd)
{
    for (auto smoothing : {0.3, 1.0, 5.0}) {
        SmoothCurve curve(createZigzag(), smoothing);

        EXPECT_NEAR(0, curve.getValues(0).at(0), 1e-9);
        EXPECT_NEAR(1, curve.getValues(0).at(1), 1e-9);
        EXPECT_NEAR(0, curve.getValues(4).at(0), 1e-9);
        EXPECT_NEAR(8, curve.getValues(4).at(1), 1e-9);
    }
}

TEST_F(SmoothCurveTests, smoothing_keepsUniformMotion)
{
    SmoothCurve curve({{.time = 0, .values = {0}}, {.time = 2, .values = {4}}, {.time = 3, .values = {6}}}, 1.0);

    EXPECT_NEAR(1, curve.getValues(0.5).at(0), 1e-9);
    EXPECT_NEAR(4, curve.getValues(2).at(0), 1e-9);
    EXPECT_NEAR(5, curve.getValues(2.5).at(0), 1e-9);
}

TEST_F(SmoothCurveTests, smoothing_roundsCorners)
{
    SmoothCurve curve(createZigzag(), 0.5);

    auto peak = curve.getValues(1).at(0);
    EXPECT_LT(peak, 10.0);
    EXPECT_GT(peak, 5.0);
    EXPECT_GT(curve.getValues(3).at(0), -5.0);
}

TEST_F(SmoothCurveTests, smoothing_matchesNumericalConvolution)
{
    SmoothCurve unsmoothedCurve(createZigzag(), 0);
    for (auto smoothing : {0.3, 1.0, 5.0}) {
        SmoothCurve curve(createZigzag(), smoothing);
        for (auto time : {0.0, 0.5, 1.0, 2.2, 3.9, 4.0}) {
            for (int channel = 0; channel < 2; ++channel) {
                EXPECT_NEAR(integrateSmoothedValue(unsmoothedCurve, channel, time, smoothing), curve.getValues(time).at(channel), 1e-3);
            }
        }
    }
}

TEST_F(SmoothCurveTests, singleKeyframe_isConstant)
{
    SmoothCurve curve({{.time = 2, .values = {3, 4}}}, 1.0);

    EXPECT_DOUBLE_EQ(3, curve.getValues(0).at(0));
    EXPECT_DOUBLE_EQ(4, curve.getValues(7).at(1));
    EXPECT_DOUBLE_EQ(2, curve.getStartTime());
    EXPECT_DOUBLE_EQ(2, curve.getEndTime());
}
