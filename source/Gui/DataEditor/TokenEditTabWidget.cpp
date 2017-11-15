#include "TokenEditTabWidget.h"

#include "TabWidgetHelper.h"
#include "TokenEditTab.h"
#include "DataEditModel.h"

TokenEditTabWidget::TokenEditTabWidget(QWidget * parent) : QTabWidget(parent)
{
	connect(this, &TokenEditTabWidget::currentChanged, [this](int index) {
		_currentIndex = index;
	});
}

void TokenEditTabWidget::init(DataEditModel * model, DataEditController * controller)
{
	_model = model;
	_controller = controller;
}

void TokenEditTabWidget::updateDisplay()
{
	auto const& cell = _model->getCellToEditRef();
	if (!cell.tokens) {
		return;
	}

	int origIndex = _currentIndex;

	clear();
	for (auto tokenTab : tokenTabs) {
		delete tokenTab;
	}
	tokenTabs.clear();

	int numToken = cell.tokens->size();
	for (int tokenIndex = 0; tokenIndex < numToken; ++tokenIndex) {
		auto tokenTab = new TokenEditTab();
		addTab(tokenTab, "token " + QString::number(tokenIndex));
		tokenTabs.push_back(tokenTab);
	}
	if (origIndex >= tokenTabs.size()) {
		origIndex = tokenTabs.size() - 1;
	}
	setCurrentIndex(origIndex);
}
