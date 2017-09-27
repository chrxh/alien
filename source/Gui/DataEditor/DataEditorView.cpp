#include <QTabWidget>

#include "Gui/Settings.h"
#include "ClusterEditWidget.h"
#include "CellEditWidget.h"
#include "DataEditorModel.h"

#include "DataEditorView.h"

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

	_clusterEditTab = new ClusterEditWidget(_mainTabWidget);
	_clusterEditTab->setFrameShape(QFrame::NoFrame);
	_clusterEditTab->setLineWidth(0);
	_clusterEditTab->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	_clusterEditTab->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	_clusterEditTab->setOverwriteMode(false);
	_clusterEditTab->setCursorWidth(6);
	_clusterEditTab->setPalette(GuiSettings::getPaletteForTab());
	_mainTabWidget->addTab(_clusterEditTab, "cluster");

	_cellEditTab = new CellEditWidget(_mainTabWidget);
	_cellEditTab->setPalette(GuiSettings::getPaletteForTab());
	_cellEditTab->setFrameShape(QFrame::NoFrame);
	_cellEditTab->setFrameShadow(QFrame::Plain);
	_cellEditTab->setCursorWidth(6);
	_mainTabWidget->addTab(_cellEditTab, "cell");

	update();
}

void DataEditorView::init(IntVector2D const & upperLeftPosition, DataEditorModel* model)
{
	_mainTabWidget->setGeometry(upperLeftPosition.x, upperLeftPosition.y, _mainTabWidget->width(), _mainTabWidget->height());
	_model = model;
}

void DataEditorView::update() const
{
	if (!_visible || _editorSelector == EditorSelector::No) {
		_mainTabWidget->setVisible(false);
		return;
	}

	if (_editorSelector == EditorSelector::Cluster) {
		_mainTabWidget->setVisible(true);
		_clusterEditTab->updateCluster(_model->selectedCluster);
		_cellEditTab->updateCell(_model->selectedCell);
	}
}

void DataEditorView::switchToNoEditor()
{
	_editorSelector = EditorSelector::No;
	update();
}

void DataEditorView::switchToClusterEditor()
{
	_editorSelector = EditorSelector::Cluster;
	update();
}

void DataEditorView::show(bool visible)
{
	_visible = visible;
	update();
}
