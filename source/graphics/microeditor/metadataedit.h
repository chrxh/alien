#ifndef METADATAEDIT_H
#define METADATAEDIT_H

#include <QWidget>

namespace Ui {
class MetaDataEdit;
}

class MetaDataEdit : public QWidget
{
    Q_OBJECT

public:
    explicit MetaDataEdit(QWidget *parent = 0);
    ~MetaDataEdit();

    void updateMetaData (QString clusterName, QString cellName, quint8 cellColor, QString cellDescription);
    void requestUpdate ();

signals:
    void metaDataChanged (QString clusterName, QString cellName, quint8 cellColor, QString cellDescription);

private slots:
    void changesFromMetaDataPropertiesEditor (QString clusterName, QString cellName, quint8 cellColor);
    void changesFromMetaDataDescriptionEditor ();

private:
    Ui::MetaDataEdit *ui;

    QString _clusterName;
    QString _cellName;
    quint8 _cellColor;
    QString _cellDescription;
};

#endif // METADATAEDIT_H
