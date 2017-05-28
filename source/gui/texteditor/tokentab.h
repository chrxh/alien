#ifndef TOKENTAB_H
#define TOKENTAB_H

#include <QWidget>
#include <QMap>

#include "Model/Definitions.h"

namespace Ui {
    class TokenTab;
}

class HexEdit;
class QSignalMapper;
class TokenTab : public QWidget
{
    Q_OBJECT
    
public:
    TokenTab(QWidget *parent = 0);
    ~TokenTab();

    void update (SymbolTable* symbolTable, qreal tokenEnergy, QByteArray const& tokenData);
    void requestUpdate ();

Q_SIGNALS:
    void tokenMemoryChanged (QByteArray data);
    void tokenPropChanged (qreal energy);

private Q_SLOTS:
    void tokenMemoryChanged_Slot (int tokenMemPointer);
    void tokenMemoryCursorReachedBeginning_Slot (int tokenMemPointer);
    void tokenMemoryCursorReachedEnd_Slot (int tokenMemPointer);

private:
    Ui::TokenTab *ui;
    QMap< quint8, HexEdit* > _hexEditList;      //associate start addresses with hex editors
    QByteArray _tokenMemory;
    QSignalMapper* _signalMapper;
    QSignalMapper* _signalMapper2;
    QSignalMapper* _signalMapper3;
};

#endif // TOKENTAB_H
