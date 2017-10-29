#include <QTabWidget>
#include <QGridLayout>

#include "Gui/Settings.h"
#include "ClusterEditTab.h"
#include "CellEditTab.h"
#include "MetadataEditTab.h"
#include "CellComputerEditTab.h"
#include "ParticleEditTab.h"

#include "DataEditorModel.h"
#include "DataEditorView.h"

namespace
{
	void setupTextEdit(QTextEdit* tab)
	{
		tab->setFrameShape(QFrame::NoFrame);
		tab->setLineWidth(0);
		tab->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
		tab->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
		tab->setOverwriteMode(false);
		tab->setCursorWidth(6);
		tab->setPalette(GuiSettings::getPaletteForTab());
	}
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

	_clusterEditTab = new ClusterEditTab(_mainTabWidget);
	setupTextEdit(_clusterEditTab);
	_clusterEditTab->setVisible(false);

	_cellEditTab = new CellEditTab(_mainTabWidget);
	setupTextEdit(_cellEditTab);
	_cellEditTab->setVisible(false);

	_metadataEditTab = new MetadataEditTab(_mainTabWidget);
	_metadataEditTab->setPalette(GuiSettings::getPaletteForTab());
	_metadataEditTab->setVisible(false);

	_particleEditTab = new ParticleEditTab(_mainTabWidget);
	setupTextEdit(_particleEditTab);
	_particleEditTab->setVisible(false);

	_computerTabWidget = new QTabWidget(parent);
	_computerTabWidget->setMinimumSize(QSize(385, 341));
	_computerTabWidget->setMaximumSize(QSize(385, 341));
	_computerTabWidget->setTabShape(QTabWidget::Triangular);
	_computerTabWidget->setElideMode(Qt::ElideNone);
	_computerTabWidget->setTabsClosable(false);
	_computerTabWidget->setPalette(GuiSettings::getPaletteForTabWidget());

	_computerEditTab = new CellComputerEditTab(_mainTabWidget);
	_computerEditTab->setPalette(GuiSettings::getPaletteForTab());
	_computerEditTab->setVisible(false);
	
	update();
}

void DataEditorView::init(IntVector2D const & upperLeftPosition, DataEditorModel* model, DataEditorController* controller, CellComputerCompiler* compiler)
{
	_model = model;
	_mainTabWidget->setGeometry(upperLeftPosition.x, upperLeftPosition.y, _mainTabWidget->width(), _mainTabWidget->height());
	_computerTabWidget->setGeometry(upperLeftPosition.x, upperLeftPosition.y + 270, _mainTabWidget->width(), _mainTabWidget->height());

	_clusterEditTab->init(_model, controller);
	_cellEditTab->init(_model, controller);
	_metadataEditTab->init(_model, controller);
	_computerEditTab->init(_model, controller, compiler);
	_particleEditTab->init(_model, controller);
}

void DataEditorView::update() const
{
	if (!_visible || _editorSelector == EditorSelector::No) {
		_mainTabWidget->setVisible(false);
		_computerTabWidget->setVisible(false);
		return;
	}

	if (_editorSelector == EditorSelector::CellWithComputer) {
		_clusterEditTab->updateDisplay();
		_cellEditTab->updateDisplay();
		_metadataEditTab->updateDisplay();
		_computerEditTab->updateDisplay();
	}

	if (_editorSelector == EditorSelector::CellWithoutComputer) {
		_clusterEditTab->updateDisplay();
		_cellEditTab->updateDisplay();
		_metadataEditTab->updateDisplay();
	}

	if (_editorSelector == EditorSelector::Particle) {
		_particleEditTab->updateDisplay();
	}
}

void DataEditorView::saveTabPositionForCellEditor()
{
	if (_editorSelector == EditorSelector::CellWithComputer || _editorSelector == EditorSelector::CellWithoutComputer) {
		_savedTabPosition = _mainTabWidget->currentIndex();
	}
}

int DataEditorView::getTabPositionForCellEditor()
{
	return _savedTabPosition;
}

void DataEditorView::switchToNoEditor()
{
	saveTabPositionForCellEditor();
	_editorSelector = EditorSelector::No;
	update();
}

void DataEditorView::switchToCellEditorWithComputer()
{
	saveTabPositionForCellEditor();

	_mainTabWidget->setVisible(true);
	_mainTabWidget->clear();
	_mainTabWidget->addTab(_clusterEditTab, "cluster");
	_mainTabWidget->addTab(_cellEditTab, "cell");
	_mainTabWidget->addTab(_metadataEditTab, "metadata");
	_mainTabWidget->setCurrentIndex(getTabPositionForCellEditor());
	_computerTabWidget->setVisible(true);
	_computerTabWidget->addTab(_computerEditTab, "cell computer");

	_editorSelector = EditorSelector::CellWithComputer;
	update();
}

void DataEditorView::switchToCellEditorWithoutComputer()
{
	saveTabPositionForCellEditor();

	_mainTabWidget->setVisible(true);
	_mainTabWidget->clear();
	_mainTabWidget->addTab(_clusterEditTab, "cluster");
	_mainTabWidget->addTab(_cellEditTab, "cell");
	_mainTabWidget->addTab(_metadataEditTab, "metadata");
	_mainTabWidget->setCurrentIndex(getTabPositionForCellEditor());
	_computerTabWidget->setVisible(false);

	_editorSelector = EditorSelector::CellWithoutComputer;
	update();
}

void DataEditorView::switchToParticleEditor()
{
	saveTabPositionForCellEditor();

	_mainTabWidget->setVisible(true);
	_computerTabWidget->setVisible(false);

	_mainTabWidget->clear();
	_mainTabWidget->addTab(_particleEditTab, "particle");

	_editorSelector = EditorSelector::Particle;
	update();
}

void DataEditorView::show(bool visible)
{
	_visible = visible;
	update();
}
