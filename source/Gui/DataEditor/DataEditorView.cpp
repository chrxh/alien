#include <QTabWidget>
#include <QGridLayout>

#include "Gui/Settings.h"
#include "ClusterEditWidget.h"
#include "CellEditWidget.h"
#include "MetadataEditWidget.h"
#include "CellComputerEditWidget.h"

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

	_metadataEditTab = new MetadataEditWidget(_mainTabWidget);
	_metadataEditTab->setPalette(GuiSettings::getPaletteForTab());
	_mainTabWidget->addTab(_metadataEditTab, "metadata");
	
	_computerTabWidget = new QTabWidget(parent);
	_computerTabWidget->setMinimumSize(QSize(385, 341));
	_computerTabWidget->setMaximumSize(QSize(385, 341));
	_computerTabWidget->setTabShape(QTabWidget::Triangular);
	_computerTabWidget->setElideMode(Qt::ElideNone);
	_computerTabWidget->setTabsClosable(false);
	_computerTabWidget->setPalette(GuiSettings::getPaletteForTabWidget());

	_computerEditTab = new CellComputerEditWidget(_mainTabWidget);
	_computerEditTab->setPalette(GuiSettings::getPaletteForTab());
	_computerTabWidget->addTab(_computerEditTab, "cell computer");

	update();
}

void DataEditorView::init(IntVector2D const & upperLeftPosition, DataEditorModel* model, DataEditorController* controller)
{
	_model = model;
	_mainTabWidget->setGeometry(upperLeftPosition.x, upperLeftPosition.y, _mainTabWidget->width(), _mainTabWidget->height());
	_computerTabWidget->setGeometry(upperLeftPosition.x, upperLeftPosition.y + 270, _mainTabWidget->width(), _mainTabWidget->height());
	_clusterEditTab->init(_model, controller);
	_cellEditTab->init(_model, controller);
	_metadataEditTab->init(_model, controller);
}

void DataEditorView::update() const
{
	if (!_visible || _editorSelector == EditorSelector::No) {
		_mainTabWidget->setVisible(false);
		_computerTabWidget->setVisible(false);
		return;
	}

	if (_editorSelector == EditorSelector::CellWithComputer) {
		_mainTabWidget->setVisible(true);
		_clusterEditTab->updateDisplay();
		_cellEditTab->updateDisplay();
		_metadataEditTab->updateDisplay();
		_computerTabWidget->setVisible(true);
	}

	if (_editorSelector == EditorSelector::CellWithoutComputer) {
		_mainTabWidget->setVisible(true);
		_clusterEditTab->updateDisplay();
		_cellEditTab->updateDisplay();
		_metadataEditTab->updateDisplay();
		_computerTabWidget->setVisible(false);
	}
}

void DataEditorView::switchToNoEditor()
{
	_editorSelector = EditorSelector::No;
	update();
}

void DataEditorView::switchToCellEditorWithoutComputer()
{
	_editorSelector = EditorSelector::CellWithoutComputer;
	update();
}

void DataEditorView::switchToCellEditorWithComputer()
{
	_editorSelector = EditorSelector::CellWithComputer;
	update();
}

void DataEditorView::show(bool visible)
{
	_visible = visible;
	update();
}
