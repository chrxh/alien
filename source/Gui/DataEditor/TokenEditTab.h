#pragma once

#include <QWidget>
#include <QMap>

#include "Model/Api/Definitions.h"
#include "Gui/Definitions.h"

namespace Ui {
    class TokenEditTab;
}

class TokenEditTab : public QWidget
{
    Q_OBJECT
    
public:
    TokenEditTab(QWidget *parent = 0);
    virtual ~TokenEditTab();

	void init(DataEditModel* model, DataEditController* controller, int tokenIndex);
	void updateDisplay();

private:
	Q_SLOT void tokenMemoryChanged_Slot (int tokenMemPointer);
	Q_SLOT void tokenMemoryCursorReachedBeginning_Slot (int tokenMemPointer);
	Q_SLOT void tokenMemoryCursorReachedEnd_Slot (int tokenMemPointer);

private:
    Ui::TokenEditTab *ui;

	DataEditModel* _model = nullptr;
	DataEditController* _controller = nullptr;
	int _tokenIndex = 0;
	
	QMap<quint8, HexEditWidget*> _hexEditByStartAddress;      //associate start addresses with hex editors
    QSignalMapper* _signalMapper;
    QSignalMapper* _signalMapper2;
    QSignalMapper* _signalMapper3;
};
