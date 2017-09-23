#include <QTableWidget>

#include "DataEditorModel.h"
#include "DataEditorView.h"

DataEditorView::DataEditorView(IntVector2D const& upperLeftPosition, DataEditorModel* model, QWidget * parent)
	: QObject(parent)
	, _upperLeftPosition(upperLeftPosition)
	, _model(model)
{
	_mainTab = new QTableWidget(parent);
	_mainTab->setGeometry(upperLeftPosition.x, upperLeftPosition.y, 100, 200);

	update();
}

void DataEditorView::update() const
{
	if (!_visible) {
		_mainTab->setVisible(false);
		return;
	}
	bool areItemsSelected = !_model->selectedCellIds.empty() || !_model->selectedParticleIds.empty();
	_mainTab->setVisible(areItemsSelected);
}

void DataEditorView::show(bool visible)
{
	_visible = visible;
	update();
}
