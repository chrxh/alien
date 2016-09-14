#ifndef TOKENTAB_H
#define TOKENTAB_H

#include <QWidget>
#include <QMap>

namespace Ui {
    class TokenTab;
}

class HexEdit;
class MetaDataManager;
class QSignalMapper;
class TokenTab : public QWidget
{
    Q_OBJECT
    
public:
    explicit TokenTab(QWidget *parent = 0);
    ~TokenTab();

    void update (qreal tokenEnergy, const QVector< quint8 >& tokenData, MetaDataManager* meta);
    void requestUpdate ();

signals:
    void tokenMemoryChanged (QVector< quint8 > data);
    void tokenPropChanged (qreal energy);

private slots:
    void tokenMemoryChanged_Slot (int tokenMemPointer);
    void tokenMemoryCursorReachedBeginning_Slot (int tokenMemPointer);
    void tokenMemoryCursorReachedEnd_Slot (int tokenMemPointer);

private:
    Ui::TokenTab *ui;
    QMap< quint8, HexEdit* > _hexEditList;      //associate start addresses with hex editors
    QVector< quint8 > _tokenMemory;
    QSignalMapper* _signalMapper;
    QSignalMapper* _signalMapper2;
    QSignalMapper* _signalMapper3;
};

#endif // TOKENTAB_H
