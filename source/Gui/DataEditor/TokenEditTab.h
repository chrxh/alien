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

	void init(DataEditModel* model, DataEditController* controller);
	void updateDisplay(int tokenIndex);

	void update(SymbolTable* symbolTable, qreal tokenEnergy, QByteArray const& tokenData);
    void requestUpdate ();

Q_SIGNALS:
    void tokenMemoryChanged (QByteArray data);
    void tokenPropChanged (qreal energy);

private Q_SLOTS:
    void tokenMemoryChanged_Slot (int tokenMemPointer);
    void tokenMemoryCursorReachedBeginning_Slot (int tokenMemPointer);
    void tokenMemoryCursorReachedEnd_Slot (int tokenMemPointer);

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
