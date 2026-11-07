#include <cmath>

#include <gtest/gtest.h>

#include <Video/CameraPath.h>

class CameraPathTests : public ::testing::Test
{};

TEST_F(CameraPathTests, withoutSmoothing_reachesWaypoints)
{
    CameraPath path(
        {
            {.center = {100.0f, 200.0f}, .zoom = 1.0f, .timeLapse = 1.0f},
            {.center = {300.0f, 200.0f}, .zoom = 4.0f, .duration = 2.0f, .timeLapse = 10.0f},
            {.center = {300.0f, 0.0f}, .zoom = 2.0f, .duration = 3.0f, .timeLapse = 1.0f},
        },
        0);

    EXPECT_FLOAT_EQ(5.0f, path.getDuration());

    auto start = path.getState(0);
    EXPECT_FLOAT_EQ(100.0f, start.center.x);
    EXPECT_FLOAT_EQ(200.0f, start.center.y);
    EXPECT_FLOAT_EQ(1.0f, start.zoom);
    EXPECT_FLOAT_EQ(1.0f, start.timeLapse);

    auto second = path.getState(2.0);
    EXPECT_FLOAT_EQ(300.0f, second.center.x);
    EXPECT_FLOAT_EQ(4.0f, second.zoom);
    EXPECT_FLOAT_EQ(10.0f, second.timeLapse);

    auto end = path.getState(5.0);
    EXPECT_FLOAT_EQ(0.0f, end.center.y);
    EXPECT_FLOAT_EQ(2.0f, end.zoom);
}

TEST_F(CameraPathTests, withoutSmoothing_interpolatesZoomAndTimeLapseGeometrically)
{
    CameraPath path(
        {
            {.center = {0.0f, 0.0f}, .zoom = 1.0f, .timeLapse = 1.0f},
            {.center = {10.0f, 0.0f}, .zoom = 4.0f, .duration = 2.0f, .timeLapse = 100.0f},
        },
        0);

    auto middle = path.getState(1.0);
    EXPECT_FLOAT_EQ(5.0f, middle.center.x);
    EXPECT_NEAR(2.0f, middle.zoom, 1.0e-5f);
    EXPECT_NEAR(10.0f, middle.timeLapse, 1.0e-4f);
}

TEST_F(CameraPathTests, smoothing_keepsFirstAndLastWaypoint)
{
    CameraPath path(
        {
            {.center = {0.0f, 0.0f}, .zoom = 1.0f},
            {.center = {50.0f, 80.0f}, .zoom = 8.0f, .duration = 1.0f},
            {.center = {100.0f, 0.0f}, .zoom = 2.0f, .duration = 1.0f},
        },
        1.0f);

    auto start = path.getState(0);
    EXPECT_NEAR(0.0f, start.center.x, 1.0e-4f);
    EXPECT_NEAR(1.0f, start.zoom, 1.0e-5f);

    auto end = path.getState(2.0);
    EXPECT_NEAR(100.0f, end.center.x, 1.0e-4f);
    EXPECT_NEAR(0.0f, end.center.y, 1.0e-4f);
    EXPECT_NEAR(2.0f, end.zoom, 1.0e-5f);

    // The corner at the middle waypoint is cut
    EXPECT_LT(path.getState(1.0).center.y, 80.0f);
}

TEST_F(CameraPathTests, tooShortDurations_areExtended)
{
    CameraPath path({{.center = {0.0f, 0.0f}}, {.center = {10.0f, 0.0f}, .duration = 0.0f}}, 0);

    EXPECT_FLOAT_EQ(CameraPath::MinDuration, path.getDuration());
}

TEST_F(CameraPathTests, singleWaypoint_staysInPlace)
{
    CameraPath path({{.center = {7.0f, 8.0f}, .zoom = 3.0f, .timeLapse = 5.0f}}, 1.0f);

    EXPECT_FLOAT_EQ(0.0f, path.getDuration());
    auto state = path.getState(0);
    EXPECT_FLOAT_EQ(7.0f, state.center.x);
    EXPECT_FLOAT_EQ(8.0f, state.center.y);
    EXPECT_NEAR(3.0f, state.zoom, 1.0e-5f);
    EXPECT_NEAR(5.0f, state.timeLapse, 1.0e-5f);
}
