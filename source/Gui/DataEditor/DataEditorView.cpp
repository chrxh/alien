#include <QTabWidget>
#include <QGridLayout>

#include "Gui/Settings.h"
#include "ClusterEditTab.h"
#include "CellEditTab.h"
#include "MetadataEditTab.h"
#include "CellComputerEditTab.h"
#include "ParticleEditTab.h"
#include "SelectionEditTab.h"
#include "SymbolEditTab.h"
#include "TokenEditTabWidget.h"

#include "DataEditModel.h"
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

	void setupTabWidget(QTabWidget* tabWidget, QSize const& size)
	{
		tabWidget->setMinimumSize(size);
		tabWidget->setMaximumSize(size);
		tabWidget->setTabShape(QTabWidget::Triangular);
		tabWidget->setElideMode(Qt::ElideNone);
		tabWidget->setTabsClosable(false);
		tabWidget->setPalette(GuiSettings::getPaletteForTabWidget());
	}
}

DataEditorView::DataEditorView(QWidget * parent)
	: QObject(parent)
{
	//main tabs
	_mainTabWidget = new QTabWidget(parent);
	setupTabWidget(_mainTabWidget, QSize(385, 260));

	_clusterTab = new ClusterEditTab(parent);
	setupTextEdit(_clusterTab);
	_clusterTab->setVisible(false);

	_cellTab = new CellEditTab(parent);
	setupTextEdit(_cellTab);
	_cellTab->setVisible(false);

	_metadataTab = new MetadataEditTab(parent);
	_metadataTab->setPalette(GuiSettings::getPaletteForTab());
	_metadataTab->setVisible(false);

	_particleTab = new ParticleEditTab(parent);
	setupTextEdit(_particleTab);
	_particleTab->setVisible(false);

	_selectionTab = new SelectionEditTab(parent);
	setupTextEdit(_selectionTab);
	_selectionTab->setVisible(false);

	//computer tabs
	_computerTabWidget = new QTabWidget(parent);
	setupTabWidget(_computerTabWidget, QSize(385, 341));

	_computerTab = new CellComputerEditTab(parent);
	_computerTab->setPalette(GuiSettings::getPaletteForTab());
	_computerTab->setVisible(false);

	//symbol tabs
	_symbolTabWidget = new QTabWidget(parent);
	setupTabWidget(_symbolTabWidget, QSize(385, 260));

	_symbolTab = new SymbolEditTab(parent);
	_symbolTab->setVisible(false);

	//token tabs
	_tokenTabWidget = new TokenEditTabWidget(parent);
	setupTabWidget(_tokenTabWidget, QSize(385, 260));

	update();
}

void DataEditorView::init(IntVector2D const & upperLeftPosition, DataEditModel* model, DataEditController* controller, CellComputerCompiler* compiler)
{
	_model = model;
	_upperLeftPosition = upperLeftPosition;
	_mainTabWidget->setGeometry(upperLeftPosition.x, upperLeftPosition.y, _mainTabWidget->width(), _mainTabWidget->height());
	_computerTabWidget->setGeometry(upperLeftPosition.x, upperLeftPosition.y + 270, _computerTabWidget->width(), _computerTabWidget->height());
	_symbolTabWidget->setGeometry(upperLeftPosition.x + 395, upperLeftPosition.y, _symbolTabWidget->width(), _symbolTabWidget->height());
	_tokenTabWidget->setGeometry(upperLeftPosition.x + 395 + _symbolTabWidget->width() + 10, upperLeftPosition.y, _tokenTabWidget->width(), _tokenTabWidget->height());

	_clusterTab->init(_model, controller);
	_cellTab->init(_model, controller);
	_metadataTab->init(_model, controller);
	_computerTab->init(_model, controller, compiler);
	_particleTab->init(_model, controller);
	_selectionTab->init(_model, controller);
	_symbolTab->init(_model, controller);
}

void DataEditorView::update() const
{
	if (!_visible || _editorSelector == EditorSelector::No) {
		_mainTabWidget->setVisible(false);
		_computerTabWidget->setVisible(false);
		_symbolTabWidget->setVisible(false);
		_tokenTabWidget->setVisible(false);
		return;
	}

	if (_editorSelector == EditorSelector::Selection) {
		_mainTabWidget->setVisible(true);
		_computerTabWidget->setVisible(false);
		_symbolTabWidget->setVisible(false);
		_tokenTabWidget->setVisible(false);

		_selectionTab->updateDisplay();
	}

	if (_editorSelector == EditorSelector::CellWithComputerWithoutToken) {
		_mainTabWidget->setVisible(true);
		_computerTabWidget->setVisible(true);
		_symbolTabWidget->setVisible(true);
		_tokenTabWidget->setVisible(false);

		_clusterTab->updateDisplay();
		_cellTab->updateDisplay();
		_metadataTab->updateDisplay();
		_computerTab->updateDisplay();
		_symbolTab->updateDisplay();
	}

	if (_editorSelector == EditorSelector::CellWithoutComputerWithoutToken) {
		_mainTabWidget->setVisible(true);
		_computerTabWidget->setVisible(false);
		_symbolTabWidget->setVisible(false);
		_tokenTabWidget->setVisible(false);

		_clusterTab->updateDisplay();
		_cellTab->updateDisplay();
		_metadataTab->updateDisplay();
	}

	if (_editorSelector == EditorSelector::Particle) {
		_mainTabWidget->setVisible(true);
		_computerTabWidget->setVisible(false);
		_symbolTabWidget->setVisible(false);
		_tokenTabWidget->setVisible(false);

		_particleTab->updateDisplay();
	}
}

void DataEditorView::saveTabPositionForCellEditor()
{
	if (_editorSelector == EditorSelector::CellWithComputerWithoutToken || _editorSelector == EditorSelector::CellWithoutComputerWithoutToken) {
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

	_mainTabWidget->clear();
	_mainTabWidget->addTab(_clusterTab, "cluster");
	_mainTabWidget->addTab(_cellTab, "cell");
	_mainTabWidget->addTab(_metadataTab, "metadata");
	_mainTabWidget->setCurrentIndex(getTabPositionForCellEditor());
	_computerTabWidget->clear();
	_computerTabWidget->addTab(_computerTab, "cell computer");
	_symbolTabWidget->clear();
	_symbolTabWidget->addTab(_symbolTab, "symbols");

	_editorSelector = EditorSelector::CellWithComputerWithoutToken;
	update();
}

void DataEditorView::switchToCellEditorWithoutComputer()
{
	saveTabPositionForCellEditor();

	_mainTabWidget->clear();
	_mainTabWidget->addTab(_clusterTab, "cluster");
	_mainTabWidget->addTab(_cellTab, "cell");
	_mainTabWidget->addTab(_metadataTab, "metadata");
	_mainTabWidget->setCurrentIndex(getTabPositionForCellEditor());

	_editorSelector = EditorSelector::CellWithoutComputerWithoutToken;
	update();
}

void DataEditorView::switchToParticleEditor()
{
	saveTabPositionForCellEditor();

	_mainTabWidget->clear();
	_mainTabWidget->addTab(_particleTab, "particle");

	_editorSelector = EditorSelector::Particle;
	update();
}

void DataEditorView::switchToSelectionEditor()
{
	saveTabPositionForCellEditor();

	_mainTabWidget->clear();
	_mainTabWidget->addTab(_selectionTab, "selection");

	_editorSelector = EditorSelector::Selection;
	update();
}

void DataEditorView::show(bool visible)
{
	_visible = visible;
	update();
}
