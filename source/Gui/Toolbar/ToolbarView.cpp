#include <QPushButton>

#include "Gui/Settings.h"

#include "ToolbarView.h"
#include "ToolbarController.h"

ToolbarView::ToolbarView(QWidget * parent)
	: QWidget(parent)
{
	ui.setupUi(this);
	setStyleSheet(BUTTON_STYLESHEET);
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
}

void ToolbarView::setEnableDeleteSelections(bool enable)
{
	ui.delSelectionButton->setEnabled(enable);
	ui.delExtendedSelectionButton->setEnabled(enable);
}

void ToolbarView::setEnableAddToken(bool enable)
{
	ui.addTokenButton->setEnabled(enable);
}

void ToolbarView::setEnableDeleteToken(bool enable)
{
	ui.delTokenButton->setEnabled(enable);
}

