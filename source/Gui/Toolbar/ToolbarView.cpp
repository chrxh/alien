#include <QPushButton>

#include "Gui/Settings.h"

#include "ToolbarView.h"
#include "ToolbarController.h"

ToolbarView::ToolbarView(QWidget * parent)
	: QWidget(parent)
{
	ui.setupUi(this);
	setStyleSheet(GuiSettings::ButtonStyleSheet);
}

void ToolbarView::init(IntVector2D const & upperLeftPosition, ToolbarController* controller)
{
	_controller = controller;
	setGeometry(upperLeftPosition.x, upperLeftPosition.y, width(), height());

	connect(ui.requestCellButton, &QPushButton::clicked, [this]() {
		_controller->onRequestCell();
	});
	connect(ui.requestParticleButton, &QPushButton::clicked, [this]() {
		_controller->onRequestParticle();
	});
	connect(ui.delSelectionButton, &QPushButton::clicked, [this]() {
		_controller->onDeleteSelection();
	});
	connect(ui.delExtendedSelectionButton, &QPushButton::clicked, [this]() {
		_controller->onDeleteExtendedSelection();
	});
	connect(ui.requestTokenButton, &QPushButton::clicked, [this]() {
		_controller->onRequestToken();
	});
	connect(ui.delTokenButton, &QPushButton::clicked, [this]() {
		_controller->onDeleteToken();
	});
	connect(ui.showCellInfoButton, &QPushButton::toggled, [this](bool checked) {
		if (checked) {
			ui.showCellInfoButton->setIcon(QIcon("://Icons/info_on.png"));
		}
		else {
			ui.showCellInfoButton->setIcon(QIcon("://Icons/info_off.png"));
		}
		_controller->onToggleCellInfo(checked);
	});
	ui.showCellInfoButton->setChecked(false);
}

void ToolbarView::setEnableDeleteSelections(bool enable)
{
	ui.delSelectionButton->setEnabled(enable);
	ui.delExtendedSelectionButton->setEnabled(enable);
}

void ToolbarView::setEnableAddToken(bool enable)
{
	ui.requestTokenButton->setEnabled(enable);
}

void ToolbarView::setEnableDeleteToken(bool enable)
{
	ui.delTokenButton->setEnabled(enable);
}

