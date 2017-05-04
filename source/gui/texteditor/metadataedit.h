#ifndef METADATAEDIT_H
#define METADATAEDIT_H

#include <QWidget>

namespace Ui {
class MetadataEdit;
}

class MetadataEdit : public QWidget
{
    Q_OBJECT

public:
    explicit MetadataEdit(QWidget *parent = 0);
    ~MetadataEdit();

    void updateMetadata (QString clusterName, QString cellName, quint8 cellColor, QString cellDescription);
    void requestUpdate ();

Q_SIGNALS:
    void metadataChanged (QString clusterName, QString cellName, quint8 cellColor, QString cellDescription);

private Q_SLOTS:
    void changesFromMetadataPropertiesEditor (QString clusterName, QString cellName, quint8 cellColor);
    void changesFromMetadataDescriptionEditor ();

private:
    Ui::MetadataEdit *ui;

    QString _clusterName;
    QString _cellName;
    quint8 _cellColor;
    QString _cellDescription;
};

#endif // METADATAEDIT_H
