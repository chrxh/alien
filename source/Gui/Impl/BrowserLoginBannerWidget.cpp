#include "BrowserLoginBannerWidget.h"

#include <string>

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include <Network/NetworkService.h>

#include "AlienGui.h"
#include "LoginDialog.h"
#include "StyleService.h"

namespace
{
    auto constexpr PaddingX = 12.0f;
    auto constexpr PaddingY = 6.0f;
    auto constexpr BarWidth = 3.0f;
    auto constexpr IconSpacing = 9.0f;
    auto constexpr TextSpacing = 12.0f;

    auto const Icon = std::string(ICON_FA_USER_PLUS);
    auto const Headline = std::string("Join the ALIEN community");
    auto const SubText = std::string("Upload your own simulations, react to those of others and get a private workspace. It's free.");
    auto const ButtonText = std::string("Log in or register");
}

BrowserLoginBannerWidget _BrowserLoginBannerWidget::create()
{
    return BrowserLoginBannerWidget(new _BrowserLoginBannerWidget());
}

void _BrowserLoginBannerWidget::process()
{
    if (!isVisible()) {
        return;
    }

    auto const& style = ImGui::GetStyle();
    auto width = ImGui::GetContentRegionAvail().x;
    auto height = ImGui::GetFrameHeight() + scale(PaddingY) * 2;

    auto pos = ImGui::GetCursorScreenPos();
    auto drawList = ImGui::GetWindowDrawList();
    drawList->AddRectFilled(pos, {pos.x + width, pos.y + height}, Const::BrowserLoginBannerColor);
    drawList->AddRectFilled(pos, {pos.x + scale(BarWidth), pos.y + height}, Const::BrowserLoginBannerBarColor);

    auto loginButtonWidth = ImGui::CalcTextSize(ButtonText.c_str()).x + style.FramePadding.x * 2;
    auto dismissButtonWidth = ImGui::CalcTextSize(ICON_FA_TIMES).x + style.FramePadding.x * 2;
    auto buttonsPosX = pos.x + width - scale(PaddingX) - loginButtonWidth - dismissButtonWidth - style.ItemSpacing.x;

    auto textPosY = pos.y + (height - ImGui::GetTextLineHeight()) / 2;
    auto textPosX = pos.x + scale(PaddingX) + scale(BarWidth);
    drawList->AddText({textPosX, textPosY}, Const::BrowserLoginBannerBarColor, Icon.c_str());

    textPosX += ImGui::CalcTextSize(Icon.c_str()).x + scale(IconSpacing);
    drawList->AddText({textPosX, textPosY}, Const::TextBaseColor, Headline.c_str());

    textPosX += ImGui::CalcTextSize(Headline.c_str()).x + scale(TextSpacing);
    if (textPosX + ImGui::CalcTextSize(SubText.c_str()).x < buttonsPosX - scale(TextSpacing)) {
        drawList->AddText({textPosX, textPosY}, Const::TextDecentColor, SubText.c_str());
    }

    ImGui::SetCursorScreenPos({buttonsPosX, pos.y + scale(PaddingY)});
    if (AlienGui::ActionButton(AlienGui::ActionButtonParameters().buttonText(ButtonText).highlighted(true).frame(true).transparentBackground(false))) {
        LoginDialog::get().open();
    }

    ImGui::SameLine();
    if (AlienGui::ActionButton(AlienGui::ActionButtonParameters().buttonText(ICON_FA_TIMES).tooltip("Hide the invitation until the next program start"))) {
        _dismissed = true;
    }

    ImGui::SetCursorScreenPos({pos.x, pos.y + height + style.ItemSpacing.y});
}

bool _BrowserLoginBannerWidget::isVisible() const
{
    return !_dismissed && !NetworkService::get().getLoggedInUserName();
}
