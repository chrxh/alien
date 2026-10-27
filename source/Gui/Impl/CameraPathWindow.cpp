#include "CameraPathWindow.h"

#include <filesystem>

#include <imgui.h>

#include <Fonts/AlienIconFont.h>
#include <Fonts/IconsFontAwesome5.h>

#include <Base/GlobalSettings.h>
#include <Base/StringHelper.h>

#include <Engine/Interface/SimulationFacade.h>

#include "AlienGui.h"
#include "CameraPathController.h"
#include "GenericFileDialog.h"
#include "GenericMessageDialog.h"
#include "SimulationView.h"
#include "StyleService.h"

namespace
{
    auto constexpr RightColumnWidth = 230.0f;
    auto constexpr DefaultSettingsHeight = 140.0f;
    auto constexpr RowHeight = 23.0f;
    auto constexpr DefaultDuration = 5.0f;

    bool inputCell(std::string const& id, float& value, char const* format)
    {
        ImGui::SetNextItemWidth(-1);
        return ImGui::InputFloat(id.c_str(), &value, 0, 0, format);
    }

    std::string formatCount(int value)
    {
        return StringHelper::format(static_cast<uint64_t>(value));
    }
}

CameraPathWindow::CameraPathWindow()
    : AlienWindow("Camera path", "windows.camera path", false, true, {60.0f, 80.0f}, {640.0f, 460.0f})
{}

void CameraPathWindow::initIntern()
{
    _settingsOpen = GlobalSettings::get().getValue("windows.camera path.settings.open", _settingsOpen);
    _settingsHeight = GlobalSettings::get().getValue("windows.camera path.settings.height", scale(DefaultSettingsHeight))
        * WindowController::get().getContentScaleCorrection();
    _videoDirectory = GlobalSettings::get().getValue("windows.camera path.video directory", std::filesystem::current_path().string());
}

void CameraPathWindow::shutdownIntern()
{
    GlobalSettings::get().setValue("windows.camera path.settings.open", _settingsOpen);
    GlobalSettings::get().setValue("windows.camera path.settings.height", _settingsHeight);
    GlobalSettings::get().setValue("windows.camera path.video directory", _videoDirectory);
}

void CameraPathWindow::processIntern()
{
    auto numWaypoints = toInt(CameraPathController::get().getWaypoints().size());
    if (_selectedWaypoint && *_selectedWaypoint >= numWaypoints) {
        _selectedWaypoint.reset();
    }

    processToolbar();

    if (ImGui::BeginChild("##child1", {0, -scale(44.0f)})) {
        if (ImGui::BeginChild("##child2", {0, _settingsOpen ? -_settingsHeight : -scale(35.0f)})) {
            processWaypointTable();
        }
        ImGui::EndChild();

        processSettings();
    }
    ImGui::EndChild();

    processStatusBar();
}

void CameraPathWindow::processToolbar()
{
    auto& controller = CameraPathController::get();
    auto busy = controller.isPlaying() || controller.isRecording();
    auto hasPath = controller.getWaypoints().size() >= 2;

    AlienGui::Toolbar(
        AlienGui::ToolbarParameters().id("CameraPath"),
        {AlienGui::ToolbarItem::createButton(AlienGui::ToolbarItemParameters()
                                                 .icon(ICON_FA_PLUS)
                                                 .name("Add waypoint")
                                                 .tooltip("Adds the current view as waypoint behind the selected one or at the end.")
                                                 .disabled(controller.isRecording())
                                                 .action([&] { onAddWaypoint(); })),
         AlienGui::ToolbarItem::createButton(AlienGui::ToolbarItemParameters()
                                                 .icon(ICON_FA_MINUS)
                                                 .name("Delete waypoint")
                                                 .disabled(controller.isRecording() || !_selectedWaypoint.has_value())
                                                 .action([&] { onDeleteWaypoint(); })),
         AlienGui::ToolbarItem::createButton(AlienGui::ToolbarItemParameters()
                                                 .icon(ICON_CLEAR)
                                                 .name("Delete all waypoints")
                                                 .disabled(controller.isRecording() || controller.getWaypoints().empty())
                                                 .action([&] {
                                                     GenericMessageDialog::get().yesNo("Delete", "Do you really want to delete all waypoints?", [&] {
                                                         CameraPathController::get().setWaypoints({});
                                                         _selectedWaypoint.reset();
                                                     });
                                                 })),
         AlienGui::ToolbarItem::createSeparator(),
         AlienGui::ToolbarItem::createButton(AlienGui::ToolbarItemParameters()
                                                 .icon(ICON_FA_PLAY)
                                                 .name("Play")
                                                 .tooltip("Flies the camera along the path in real time. The time-lapse only applies to the video.")
                                                 .disabled(busy || !hasPath)
                                                 .action([&] { CameraPathController::get().startPlayback(); })),
         AlienGui::ToolbarItem::createButton(AlienGui::ToolbarItemParameters()
                                                 .icon(ICON_FA_VIDEO)
                                                 .name("Record video")
                                                 .tooltip("Flies the camera along the path and records the simulation without the user interface as MP4 video. "
                                                          "The video has the size of the view.")
                                                 .disabled(busy || !hasPath)
                                                 .action([&] { onRecordVideo(); })),
         AlienGui::ToolbarItem::createButton(AlienGui::ToolbarItemParameters()
                                                 .icon(ICON_FA_STOP)
                                                 .name("Stop")
                                                 .tooltip("Stops the camera flight. A recorded video keeps the frames up to this point.")
                                                 .disabled(!busy)
                                                 .action([&] { CameraPathController::get().stop(); }))});
}

void CameraPathWindow::processWaypointTable()
{
    auto& controller = CameraPathController::get();
    auto waypoints = controller.getWaypoints();
    auto changed = false;

    static ImGuiTableFlags flags = ImGuiTableFlags_Resizable | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV
        | ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX;

    ImGui::BeginDisabled(controller.isRecording());
    if (ImGui::BeginTable("Waypoints", 7, flags, ImVec2(-1, -1), 0.0f)) {
        ImGui::TableSetupColumn("Waypoint", ImGuiTableColumnFlags_WidthFixed, scale(70.0f));
        ImGui::TableSetupColumn("Center x", ImGuiTableColumnFlags_WidthFixed, scale(80.0f));
        ImGui::TableSetupColumn("Center y", ImGuiTableColumnFlags_WidthFixed, scale(80.0f));
        ImGui::TableSetupColumn("Zoom", ImGuiTableColumnFlags_WidthFixed, scale(65.0f));
        ImGui::TableSetupColumn("Duration", ImGuiTableColumnFlags_WidthFixed, scale(75.0f));
        ImGui::TableSetupColumn("Time-lapse", ImGuiTableColumnFlags_WidthFixed, scale(80.0f));
        ImGui::TableSetupColumn("View", ImGuiTableColumnFlags_WidthFixed, scale(60.0f));
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableHeadersRow();
        ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, Const::TableHeaderColor);

        for (int row = 0; row < toInt(waypoints.size()); ++row) {
            auto& waypoint = waypoints.at(row);

            ImGui::PushID(row);
            ImGui::TableNextRow(0, scale(RowHeight));

            ImGui::TableNextColumn();
            auto selected = _selectedWaypoint == row;
            if (AlienGui::TableRowSelectable("", &selected, ImGuiSelectableFlags_None, RealVector2D(0, scale(RowHeight) - ImGui::GetStyle().FramePadding.y))) {
                _selectedWaypoint = selected ? std::make_optional(row) : std::nullopt;
            }
            ImGui::SameLine();
            AlienGui::Text(std::to_string(row + 1));

            ImGui::TableNextColumn();
            changed |= inputCell("##centerX", waypoint.center.x, "%.1f");

            ImGui::TableNextColumn();
            changed |= inputCell("##centerY", waypoint.center.y, "%.1f");

            ImGui::TableNextColumn();
            changed |= inputCell("##zoom", waypoint.zoom, "%.2f");

            // The duration leads from the previous waypoint to this one
            ImGui::TableNextColumn();
            if (row > 0) {
                changed |= inputCell("##duration", waypoint.duration, "%.1f s");
            } else {
                AlienGui::Text(AlienGui::TextParameters().text("Start").style(AlienGui::TextStyle::Decent));
            }

            ImGui::TableNextColumn();
            changed |= inputCell("##timeLapse", waypoint.timeLapse, "%.2fx");

            ImGui::TableNextColumn();
            if (AlienGui::ActionButton(AlienGui::ActionButtonParameters().buttonText(ICON_FA_EYE))) {
                controller.moveViewToWaypoint(waypoint);
            }
            AlienGui::Tooltip(AlienGui::TooltipParameters().text("Shows the waypoint").delay(false));
            ImGui::SameLine();
            if (AlienGui::ActionButton(AlienGui::ActionButtonParameters().buttonText(ICON_FA_CROSSHAIRS))) {
                auto view = controller.createWaypointFromView();
                waypoint.center = view.center;
                waypoint.zoom = view.zoom;
                changed = true;
            }
            AlienGui::Tooltip(AlienGui::TooltipParameters().text("Takes over the current view").delay(false));

            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    ImGui::EndDisabled();

    if (changed) {
        controller.setWaypoints(waypoints);
    }
}

void CameraPathWindow::processSettings()
{
    ImGui::Spacing();
    ImGui::Spacing();
    if (_settingsOpen) {
        AlienGui::MovableHorizontalSeparator(AlienGui::MovableHorizontalSeparatorParameters().additive(false), _settingsHeight);
    }

    _settingsOpen = AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name("Settings").rank(AlienGui::TreeNodeRank::High).defaultOpen(_settingsOpen));
    if (_settingsOpen) {
        if (ImGui::BeginChild("##cameraPathSettings", {scale(0), 0})) {
            auto& controller = CameraPathController::get();
            ImGui::BeginDisabled(controller.isRecording());

            auto smoothing = controller.getSmoothing();
            if (AlienGui::SliderFloat(
                    AlienGui::SliderFloatParameters()
                        .name("Smoothing")
                        .textWidth(RightColumnWidth)
                        .min(0)
                        .max(10.0f)
                        .format("%.2f s")
                        .defaultValue(&CameraPathController::DefaultSmoothing)
                        .tooltip("The path through the waypoints is smoothed with a Gaussian kernel. Its width is given in seconds. The larger it is, "
                                 "the more the path cuts the corners at the waypoints. The first and the last waypoint are always reached."),
                    &smoothing)) {
                controller.setSmoothing(smoothing);
            }

            auto fps = controller.getFps();
            if (AlienGui::InputInt(
                    AlienGui::InputIntParameters().name("Frames per second").textWidth(RightColumnWidth).defaultValue(CameraPathController::DefaultFps),
                    fps)) {
                controller.setFps(fps);
            }

            auto bitRate = controller.getBitRate();
            if (AlienGui::InputInt(
                    AlienGui::InputIntParameters()
                        .name("Bit rate (MBit/s)")
                        .textWidth(RightColumnWidth)
                        .defaultValue(CameraPathController::DefaultBitRate)
                        .tooltip("Higher bit rates keep more details of the simulation in the video but produce larger files."),
                    bitRate)) {
                controller.setBitRate(bitRate);
            }

            auto timestepsPerSecond = controller.getTimestepsPerSecond();
            if (AlienGui::InputInt(
                    AlienGui::InputIntParameters()
                        .name("Time steps per second at 1x")
                        .textWidth(RightColumnWidth)
                        .defaultValue(CameraPathController::DefaultTimestepsPerSecond)
                        .tooltip("Speed of the simulation in the video at a time-lapse of 1x. A time-lapse of 10x shows ten times as many time steps per "
                                 "second while the camera moves as fast as without time-lapse. The time-lapse is set per waypoint and changes "
                                 "smoothly on the way to the next one."),
                    timestepsPerSecond)) {
                controller.setTimestepsPerSecond(timestepsPerSecond);
            }

            ImGui::EndDisabled();
        }
        ImGui::EndChild();
    }
    AlienGui::EndTreeNode();
}

void CameraPathWindow::processStatusBar()
{
    auto& controller = CameraPathController::get();

    std::vector<std::string> statusItems;
    std::optional<std::string> warningItem;
    if (auto progress = controller.getRecordingProgress()) {
        statusItems.emplace_back("Recording frame " + formatCount(progress->frame + 1) + " of " + formatCount(progress->numFrames));
        statusItems.emplace_back(std::to_string(progress->resolution.x) + " x " + std::to_string(progress->resolution.y));
        if (!SimulationView::get().isRenderSimulation()) {
            warningItem = "The simulation is not rendered";
        }
    } else if (auto playbackTime = controller.getPlaybackTime()) {
        statusItems.emplace_back(
            "Playing " + StringHelper::format(*playbackTime, 1) + " s of " + StringHelper::format(controller.createPath().getDuration(), 1) + " s");
    } else {
        auto numWaypoints = controller.getWaypoints().size();
        statusItems.emplace_back(std::to_string(numWaypoints) + (numWaypoints == 1 ? " waypoint" : " waypoints"));
        if (numWaypoints >= 2) {
            updateVideoInfo();
            statusItems.emplace_back(StringHelper::format(controller.createPath().getDuration(), 1) + " s");
            statusItems.emplace_back(formatCount(_videoInfo->numFrames) + " frames");
            statusItems.emplace_back(StringHelper::format(_videoInfo->numTimesteps) + " time steps");
        }
    }

    AlienGui::StatusBar(AlienGui::StatusBarParameters().infoItems(statusItems).warningItem(warningItem));
}

void CameraPathWindow::onAddWaypoint()
{
    auto& controller = CameraPathController::get();
    auto waypoints = controller.getWaypoints();
    auto index = _selectedWaypoint ? *_selectedWaypoint + 1 : toInt(waypoints.size());

    // The new waypoint continues the time-lapse of its predecessor
    auto waypoint = controller.createWaypointFromView();
    waypoint.duration = index >= 2 ? waypoints.at(index - 1).duration : DefaultDuration;
    if (index >= 1) {
        waypoint.timeLapse = waypoints.at(index - 1).timeLapse;
    }
    waypoints.insert(waypoints.begin() + index, waypoint);
    controller.setWaypoints(waypoints);
    _selectedWaypoint = index;
}

void CameraPathWindow::onDeleteWaypoint()
{
    auto& controller = CameraPathController::get();
    auto waypoints = controller.getWaypoints();
    waypoints.erase(waypoints.begin() + *_selectedWaypoint);
    controller.setWaypoints(waypoints);
    if (*_selectedWaypoint >= toInt(waypoints.size())) {
        _selectedWaypoint = waypoints.empty() ? std::nullopt : std::make_optional(toInt(waypoints.size()) - 1);
    }
}

void CameraPathWindow::onRecordVideo()
{
    GenericFileDialog::get().showSaveFileDialog("Record video", "Video (*.mp4){.mp4},.*", _videoDirectory, [this](std::filesystem::path const& path) {
        auto filename = path;
        if (filename.extension().empty()) {
            filename.replace_extension(".mp4");
        }
        _videoDirectory = std::filesystem::path(filename).remove_filename().string();
        try {
            CameraPathController::get().startRecording(filename);
        } catch (AlienException const& exception) {
            GenericMessageDialog::get().information("Record video", std::string("The video could not be recorded.\n\n") + exception.what());
        }
    });
}

void CameraPathWindow::updateVideoInfo()
{
    auto& controller = CameraPathController::get();
    auto isUpToDate = _videoInfo && _videoInfo->waypoints == controller.getWaypoints() && _videoInfo->smoothing == controller.getSmoothing()
        && _videoInfo->fps == controller.getFps() && _videoInfo->timestepsPerSecond == controller.getTimestepsPerSecond();

    // While a value is edited, the previous key figures are kept, so that typing or dragging does not recalculate a long schedule each frame
    if (isUpToDate || (_videoInfo && ImGui::IsAnyItemActive())) {
        return;
    }
    auto schedule = controller.createRecordingSchedule();
    _videoInfo = VideoInfo{
        .waypoints = controller.getWaypoints(),
        .smoothing = controller.getSmoothing(),
        .fps = controller.getFps(),
        .timestepsPerSecond = controller.getTimestepsPerSecond(),
        .numFrames = schedule.getNumFrames(),
        .numTimesteps = schedule.getTotalTimesteps(),
    };
}
