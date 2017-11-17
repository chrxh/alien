#pragma once

#include <QTabWidget>

#include "Gui/Definitions.h"

class TokenEditTabWidget : public QTabWidget {
	Q_OBJECT

public:
	TokenEditTabWidget(QWidget * parent = nullptr);
	virtual ~TokenEditTabWidget() = default;

	void init(DataEditModel* model, DataEditController* controller);
	void updateDisplay();

private:
	TokenEditTab* createNewTab(int index) const;
	void deleteAllTabs();

	DataEditModel* _model = nullptr;
	DataEditController* _controller = nullptr;

	vector<TokenEditTab*> _tokenTabs;
};
