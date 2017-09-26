#include <QTabWidget>

#include "Gui/Settings.h"
#include "ClusterEditWidget.h"
#include "CellEditWidget.h"

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
	_mainTabWidget->addTab(_cellEditTab, "cell");

	update();
}

void DataEditorView::init(IntVector2D const & upperLeftPosition)
{
	_mainTabWidget->setGeometry(upperLeftPosition.x, upperLeftPosition.y, _mainTabWidget->width(), _mainTabWidget->height());
}

void DataEditorView::update() const
{
	if (!_visible || _editorSelector == EditorSelector::No) {
		_mainTabWidget->setVisible(false);
		return;
	}

	if (_editorSelector == EditorSelector::Cluster) {
		_mainTabWidget->setVisible(true);
		_clusterEditTab->updateCluster(_selectedData.clusters->at(0));
	}
}

void DataEditorView::switchToNoEditor()
{
	_editorSelector = EditorSelector::No;
	update();
}

void DataEditorView::switchToClusterEditor(ClusterDescription const & cluster)
{
	_editorSelector = EditorSelector::Cluster;
	_selectedData.clear();
	_selectedData.addCluster(cluster);
	update();
}

void DataEditorView::show(bool visible)
{
	_visible = visible;
	update();
}
