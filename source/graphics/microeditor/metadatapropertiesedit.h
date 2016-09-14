#ifndef METADATAPROPERTIESEDIT_H
#define METADATAPROPERTIESEDIT_H

#include <QTextEdit>

class MetaDataPropertiesEdit : public QTextEdit
{
    Q_OBJECT
public:
    MetaDataPropertiesEdit(QWidget *parent = 0);

    void updateMetaData (QString clusterName, QString cellName, quint8 cellColor);
    void requestUpdate ();

signals:
    void metaDataPropertiesChanged (QString clusterName, QString cellName, quint8 cellColor);

private slots:
    void keyPressEvent (QKeyEvent* e);
    void mousePressEvent (QMouseEvent* e);
    void mouseDoubleClickEvent (QMouseEvent* e);

private:
    void updateDisplay ();

    QString _clusterName;
    QString _cellName;
    quint8 _cellColor;
};

#endif // METADATAPROPERTIESEDIT_H
