#include "TokenEditTabWidget.h"

#include "TabWidgetHelper.h"
#include "TokenEditTab.h"
#include "DataEditController.h"
#include "DataEditModel.h"

TokenEditTabWidget::TokenEditTabWidget(QWidget * parent) : QTabWidget(parent)
{
}

void TokenEditTabWidget::init(DataEditModel * model, DataEditController * controller)
{
	_model = model;
	_controller = controller;

	deleteAllTabs();

	connect(this, &TokenEditTabWidget::currentChanged, [this](int index) {
		if (-1 != index) {
			_model->setSelectedTokenIndex(index);
		}
	});
}

void TokenEditTabWidget::updateDisplay()
{
	auto const& cell = _model->getCellToEditRef();
	if (!cell || !cell->tokens) {
		deleteAllTabs();
		return;
	}

	int numToken = cell->tokens->size();
	if (_tokenTabs.size() != numToken) {

		boost::optional<uint> origIndex = _model->getSelectedTokenIndex();
		deleteAllTabs();

		for (int tokenIndex = 0; tokenIndex < numToken; ++tokenIndex) {
			auto tokenTab = createNewTab(tokenIndex);
			addTab(tokenTab, "token " + QString::number(tokenIndex + 1));
			_tokenTabs.push_back(tokenTab);
		}
		if (origIndex) {
			if (*origIndex >= _tokenTabs.size()) {
				origIndex = _tokenTabs.size() - 1;
			}
			setCurrentIndex(*origIndex);
		}
	}
	for (auto tokenTab : _tokenTabs) {
		tokenTab->updateDisplay();
	}
}

TokenEditTab * TokenEditTabWidget::createNewTab(int index) const
{
	auto tokenTab = new TokenEditTab();
	tokenTab->init(_model, _controller, index);
	return tokenTab;
}

void TokenEditTabWidget::deleteAllTabs()
{
	clear();
	for (auto tokenTab : _tokenTabs) {
		delete tokenTab;
	}
	_tokenTabs.clear();
}
