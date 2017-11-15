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
	DataEditModel* _model = nullptr;
	DataEditController* _controller = nullptr;

	vector<TokenEditTab*> tokenTabs;
	int _currentIndex = 0;
};
