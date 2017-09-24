#include <QTabWidget>

#include "Gui/Settings.h"

#include "DataEditorView.h"

namespace
{
}

DataEditorView::DataEditorView(QWidget * parent)
	: QObject(parent)
{
	_mainTabWidget = new QTabWidget(parent);
	_mainTabWidget->setMinimumSize(QSize(385, 260));
	_mainTabWidget->setMaximumSize(QSize(385, 260));
	_mainTabWidget->setTabShape(QTabWidget::Triangular);
	_mainTabWidget->setElideMode(Qt::ElideNone);
	_mainTabWidget->setTabsClosable(false);
	_mainTabWidget->setPalette(GuiSettings::getPaletteForTabWidget());

	update();
}

void DataEditorView::init(IntVector2D const & upperLeftPosition)
{
	_upperLeftPosition = upperLeftPosition;
	_mainTabWidget->setGeometry(upperLeftPosition.x, upperLeftPosition.y, _mainTabWidget->width(), _mainTabWidget->height());
}

void DataEditorView::update() const
{
	if (!_visible) {
		_mainTabWidget->setVisible(false);
		return;
	}
/*
	bool areItemsSelected = !_model->selectedCellIds.empty() || !_model->selectedParticleIds.empty();
	_mainTabWidget->setVisible(areItemsSelected);
	
	enum class MainTabViewSelector { Cell, Particle, Ensemble }	selector;
	if (_model->selectedCellIds.size() + _model->selectedParticleIds.size() == 1) {
		if (!_model->selectedCellIds.empty()) {
			selector = MainTabViewSelector::Cell;
		}
		if (!_model->selectedParticleIds.empty()) {
			selector = MainTabViewSelector::Particle;
		}
	}
	else {
		selector = MainTabViewSelector::Ensemble;
	}
*/
}

void DataEditorView::show(bool visible)
{
	_visible = visible;
	update();
}
