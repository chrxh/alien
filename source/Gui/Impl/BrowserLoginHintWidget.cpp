#include "BrowserLoginHintWidget.h"

#include <algorithm>
#include <string>
#include <cfloat>

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include "AlienGui.h"
#include "LoginDialog.h"
#include "StyleService.h"

namespace
{
    auto constexpr MinCardWidth = 400.0f;
    auto constexpr CardHeight = 210.0f;
    auto constexpr CardPadding = 30.0f;
    auto constexpr CardRounding = 3.0f;

    auto constexpr IconFontSize = 44.0f;
    auto constexpr IconOffsetY = 28.0f;
    auto constexpr HeadlineOffsetY = 92.0f;
    auto constexpr SubTextOffsetY = 132.0f;
    auto constexpr ButtonOffsetY = 162.0f;

    auto const Icon = std::string(ICON_FA_LOCK);
    auto const Headline = std::string("Your private workspace");
    auto const SubText = std::string("Log in to see the simulations and genomes you uploaded.");
    auto const ButtonText = std::string("Login or register");
}

BrowserLoginHintWidget _BrowserLoginHintWidget::create()
{
    return BrowserLoginHintWidget(new _BrowserLoginHintWidget());
}

void _BrowserLoginHintWidget::process(RealVector2D const& pos, RealVector2D const& size)
{
    auto cursorPosAfterWidget = ImGui::GetCursorScreenPos();

    ImGui::SetCursorScreenPos({pos.x, pos.y});

    // Without padding the overlay covers the entire view
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    auto isVisible = ImGui::BeginChild("##loginHint", {size.x, size.y}, 0, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::PopStyleVar();

    if (isVisible) {
        AlienGui::DisabledField();
        processCard();
    }
    ImGui::EndChild();

    ImGui::SetCursorScreenPos(cursorPosAfterWidget);
}

void _BrowserLoginHintWidget::processCard()
{
    auto cardWidth = std::max(scale(MinCardWidth), ImGui::CalcTextSize(SubText.c_str()).x + scale(CardPadding) * 2);
    auto cardHeight = scale(CardHeight);

    auto windowSize = ImGui::GetWindowSize();
    auto cardPos = ImVec2{std::max(0.0f, (windowSize.x - cardWidth) / 2), std::max(0.0f, (windowSize.y - cardHeight) / 2)};

    auto windowPos = ImGui::GetWindowPos();
    auto cardScreenPos = ImVec2{windowPos.x + cardPos.x, windowPos.y + cardPos.y};
    auto cardScreenEndPos = ImVec2{cardScreenPos.x + cardWidth, cardScreenPos.y + cardHeight};

    auto drawList = ImGui::GetWindowDrawList();
    drawList->AddRectFilled(cardScreenPos, cardScreenEndPos, Const::BrowserLoginHintCardColor, scale(CardRounding));
    drawList->AddRect(cardScreenPos, cardScreenEndPos, Const::BrowserLoginHintCardBorderColor, scale(CardRounding));

    auto iconFont = StyleService::get().getIconFont();
    auto iconFontSize = scale(IconFontSize);
    auto iconWidth = iconFont->CalcTextSizeA(iconFontSize, FLT_MAX, 0.0f, Icon.c_str()).x;
    ImVec4 clipRect(cardScreenPos.x, cardScreenPos.y, cardScreenEndPos.x, cardScreenEndPos.y);
    iconFont->RenderText(
        drawList,
        iconFontSize,
        {cardScreenPos.x + (cardWidth - iconWidth) / 2, cardScreenPos.y + scale(IconOffsetY)},
        Const::BrowserLoginHintIconColor,
        clipRect,
        Icon.c_str(),
        Icon.c_str() + Icon.size(),
        0.0f,
        false);

    ImGui::PushFont(StyleService::get().getMediumBoldFont());
    auto headlineWidth = ImGui::CalcTextSize(Headline.c_str()).x;
    ImGui::SetCursorPos({cardPos.x + (cardWidth - headlineWidth) / 2, cardPos.y + scale(HeadlineOffsetY)});
    ImGui::TextUnformatted(Headline.c_str());
    ImGui::PopFont();

    auto subTextWidth = ImGui::CalcTextSize(SubText.c_str()).x;
    ImGui::SetCursorPos({cardPos.x + (cardWidth - subTextWidth) / 2, cardPos.y + scale(SubTextOffsetY)});
    AlienGui::Text(AlienGui::TextParameters().text(SubText).style(AlienGui::TextStyle::Decent));

    auto buttonWidth = ImGui::CalcTextSize(ButtonText.c_str()).x + ImGui::GetStyle().FramePadding.x * 2;
    ImGui::SetCursorPos({cardPos.x + (cardWidth - buttonWidth) / 2, cardPos.y + scale(ButtonOffsetY)});
    if (AlienGui::ActionButton(AlienGui::ActionButtonParameters().buttonText(ButtonText).highlighted(true).frame(true).transparentBackground(false))) {
        LoginDialog::get().open();
    }
}
