#include <QTabWidget>
#include <QGridLayout>

#include "Base/DebugMacros.h"

#include "Settings.h"
#include "ClusterEditTab.h"
#include "CellEditTab.h"
#include "MetadataEditTab.h"
#include "CellComputerEditTab.h"
#include "ParticleEditTab.h"
#include "SelectionEditTab.h"
#include "SymbolEditTab.h"
#include "TokenEditTabWidget.h"
#include "TabWidgetHelper.h"

#include "DataEditModel.h"
#include "DataEditView.h"


DataEditView::DataEditView(QWidget * parent)
	: QObject(parent)
{
    TRY;

	//main tabs
	_mainTabWidget = new QTabWidget(parent);
	TabWidgetHelper::setupTabWidget(_mainTabWidget, QSize(385, 260));

	_clusterTab = new ClusterEditTab(parent);
	TabWidgetHelper::setupTextEdit(_clusterTab);
	_clusterTab->setVisible(false);

	_cellTab = new CellEditTab(parent);
	TabWidgetHelper::setupTextEdit(_cellTab);
	_cellTab->setVisible(false);

	_metadataTab = new MetadataEditTab(parent);
	_metadataTab->setPalette(GuiSettings::getPaletteForTab());
	_metadataTab->setVisible(false);

	_particleTab = new ParticleEditTab(parent);
	TabWidgetHelper::setupTextEdit(_particleTab);
	_particleTab->setVisible(false);

	_selectionTab = new SelectionEditTab(parent);
	TabWidgetHelper::setupTextEdit(_selectionTab);
	_selectionTab->setVisible(false);

	//computer tabs
	_computerTabWidget = new QTabWidget(parent);
	TabWidgetHelper::setupTabWidget(_computerTabWidget, QSize(385, 341));

	_computerTab = new CellComputerEditTab(parent);
	_computerTab->setPalette(GuiSettings::getPaletteForTab());
	_computerTab->setVisible(false);

	//symbol tabs
	_symbolTabWidget = new QTabWidget(parent);
	TabWidgetHelper::setupTabWidget(_symbolTabWidget, QSize(385, 260));

	_symbolTab = new SymbolEditTab(parent);
	_symbolTab->setVisible(false);

	//token tabs
	_tokenTabWidget = new TokenEditTabWidget(parent);
	TabWidgetHelper::setupTabWidget(_tokenTabWidget, QSize(385, 260));

	updateDisplay();

	CATCH;
}

void DataEditView::init(IntVector2D const & upperLeftPosition, DataEditModel* model, DataEditController* controller, CellComputerCompiler* compiler)
{
    TRY;

	_model = model;
	_upperLeftPosition = upperLeftPosition;
	_mainTabWidget->setGeometry(upperLeftPosition.x, upperLeftPosition.y, _mainTabWidget->width(), _mainTabWidget->height());
	_computerTabWidget->setGeometry(upperLeftPosition.x, upperLeftPosition.y + _mainTabWidget->height() + 10, _computerTabWidget->width(), _computerTabWidget->height());
	_symbolTabWidget->setGeometry(upperLeftPosition.x + _mainTabWidget->width() + 10, upperLeftPosition.y, _symbolTabWidget->width(), _symbolTabWidget->height());
	_tokenTabWidget->setGeometry(_upperLeftPosition.x + _mainTabWidget->width() + 10 + _symbolTabWidget->width() + 10, _upperLeftPosition.y, _tokenTabWidget->width(), _tokenTabWidget->height());

	_clusterTab->init(_model, controller);
	_cellTab->init(_model, controller);
	_metadataTab->init(_model, controller);
	_computerTab->init(_model, controller, compiler);
	_particleTab->init(_model, controller);
	_selectionTab->init(_model, controller);
	_symbolTab->init(_model, controller);
	_tokenTabWidget->init(_model, controller);

	CATCH;
}

void DataEditView::updateDisplay(UpdateDescription update) const
{
    TRY;

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

	if (_editorSelector == EditorSelector::CellWithComputerWithToken) {
		_mainTabWidget->setVisible(true);
		_computerTabWidget->setVisible(true);
		_symbolTabWidget->setVisible(true);
		_tokenTabWidget->setVisible(true);
		_tokenTabWidget->setGeometry(_upperLeftPosition.x + _computerTabWidget->width() + 10 + _symbolTabWidget->width() + 10, _upperLeftPosition.y, _tokenTabWidget->width(), _tokenTabWidget->height());

		_clusterTab->updateDisplay();
		_cellTab->updateDisplay();
		_metadataTab->updateDisplay();
		_computerTab->updateDisplay();
		if (update != UpdateDescription::AllExceptSymbols) {
			_symbolTab->updateDisplay();
		}
		if (update != UpdateDescription::AllExceptToken) {
			_tokenTabWidget->updateDisplay();
		}
	}

	if (_editorSelector == EditorSelector::CellWithoutComputerWithToken) {
		_mainTabWidget->setVisible(true);
		_computerTabWidget->setVisible(false);
		_symbolTabWidget->setVisible(false);
		_tokenTabWidget->setVisible(true);
		_tokenTabWidget->setGeometry(_upperLeftPosition.x + _computerTabWidget->width() + 10, _upperLeftPosition.y, _tokenTabWidget->width(), _tokenTabWidget->height());

		_clusterTab->updateDisplay();
		_cellTab->updateDisplay();
		_metadataTab->updateDisplay();
		if (update != UpdateDescription::AllExceptToken) {
			_tokenTabWidget->updateDisplay();
		}
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

	CATCH;
}

void DataEditView::saveTabPositionForCellEditor()
{
    TRY;
    
	if (_editorSelector == EditorSelector::CellWithComputerWithToken
		|| _editorSelector == EditorSelector::CellWithComputerWithoutToken
		|| _editorSelector == EditorSelector::CellWithoutComputerWithToken
		|| _editorSelector == EditorSelector::CellWithoutComputerWithoutToken) {
		_savedTabPosition = _mainTabWidget->currentIndex();
	}

	CATCH;
}

int DataEditView::getTabPositionForCellEditor()
{
	return _savedTabPosition;
}

void DataEditView::switchToNoEditor()
{
    TRY;

	saveTabPositionForCellEditor();
	_editorSelector = EditorSelector::No;
	updateDisplay();

	CATCH;
}

void DataEditView::switchToCellEditorWithComputerWithToken(UpdateDescription update)
{
    TRY;

	saveTabPositionForCellEditor();

	if (_editorSelector != EditorSelector::CellWithComputerWithToken) {
		_mainTabWidget->clear();
		_mainTabWidget->addTab(_clusterTab, "cluster");
		_mainTabWidget->addTab(_cellTab, "cell");
		_mainTabWidget->addTab(_metadataTab, "metadata");
		_mainTabWidget->setCurrentIndex(getTabPositionForCellEditor());
		_computerTabWidget->clear();
		_computerTabWidget->addTab(_computerTab, "cell computer");
		if (update != UpdateDescription::AllExceptToken) {
			_symbolTabWidget->clear();
			_symbolTabWidget->addTab(_symbolTab, "symbols");
		}
		_editorSelector = EditorSelector::CellWithComputerWithToken;
	}
	updateDisplay(update);

	CATCH;
}

void DataEditView::switchToCellEditorWithoutComputerWithToken(UpdateDescription update)
{
    TRY;

	saveTabPositionForCellEditor();

	if (_editorSelector != EditorSelector::CellWithoutComputerWithToken) {
		_mainTabWidget->clear();
		_mainTabWidget->addTab(_clusterTab, "cluster");
		_mainTabWidget->addTab(_cellTab, "cell");
		_mainTabWidget->addTab(_metadataTab, "metadata");
		_mainTabWidget->setCurrentIndex(getTabPositionForCellEditor());
		_editorSelector = EditorSelector::CellWithoutComputerWithToken;
	}
	updateDisplay(update);

	CATCH;
}

void DataEditView::switchToCellEditorWithComputerWithoutToken()
{
    TRY;

	saveTabPositionForCellEditor();

	if (_editorSelector != EditorSelector::CellWithComputerWithoutToken) {
		_mainTabWidget->clear();
		_mainTabWidget->addTab(_clusterTab, "cluster");
		_mainTabWidget->addTab(_cellTab, "cell");
		_mainTabWidget->addTab(_metadataTab, "metadata");
		_mainTabWidget->setCurrentIndex(getTabPositionForCellEditor());
		_computerTabWidget->clear();
		_computerTabWidget->addTab(_computerTab, "cell computer");
		_symbolTabWidget->clear();
		_symbolTabWidget->addTab(_symbolTab, "symbol map");
		_editorSelector = EditorSelector::CellWithComputerWithoutToken;
	}
	updateDisplay();

	CATCH;
}

void DataEditView::switchToCellEditorWithoutComputerWithoutToken()
{
    TRY;
    
	saveTabPositionForCellEditor();
	if (_editorSelector != EditorSelector::CellWithoutComputerWithoutToken) {
		_mainTabWidget->clear();
		_mainTabWidget->addTab(_clusterTab, "cluster");
		_mainTabWidget->addTab(_cellTab, "cell");
		_mainTabWidget->addTab(_metadataTab, "metadata");
		_mainTabWidget->setCurrentIndex(getTabPositionForCellEditor());
		_editorSelector = EditorSelector::CellWithoutComputerWithoutToken;
	}
	updateDisplay();

	CATCH;
}

void DataEditView::switchToParticleEditor()
{
    TRY;

	saveTabPositionForCellEditor();
	if (_editorSelector != EditorSelector::Particle) {
		_mainTabWidget->clear();
		_mainTabWidget->addTab(_particleTab, "particle");
		_editorSelector = EditorSelector::Particle;
	}
	updateDisplay();

	CATCH;
}

void DataEditView::switchToSelectionEditor()
{
    TRY;

	saveTabPositionForCellEditor();
	if (_editorSelector != EditorSelector::Selection) {
		_mainTabWidget->clear();
		_mainTabWidget->addTab(_selectionTab, "selection");
		_editorSelector = EditorSelector::Selection;
	}
	updateDisplay();

	CATCH;
}

void DataEditView::show(bool visible)
{
    TRY;

	_visible = visible;
	updateDisplay();

	CATCH;
}
