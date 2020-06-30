#pragma once

#include <QWidget>
#include <QMap>

#include "ModelBasic/Definitions.h"
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
	HexEditWidget* createHexEditWidget(int size, int row, int tokenMemPointer);

	Q_SLOT void tokenMemoryChanged (int tokenMemPointer);
	Q_SLOT void tokenMemoryCursorReachedBeginning (int tokenMemPointer);
	Q_SLOT void tokenMemoryCursorReachedEnd (int tokenMemPointer);

    Ui::TokenEditTab *ui;

	DataEditModel* _model = nullptr;
	DataEditController* _controller = nullptr;
	int _tokenIndex = 0;
	
	QMap<quint8, HexEditWidget*> _hexEditByStartAddress;      //associate start addresses with hex editors
    QSignalMapper* _signalMapper;
    QSignalMapper* _signalMapper2;
    QSignalMapper* _signalMapper3;
};
