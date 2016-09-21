#ifndef METADATAPROPERTIESEDIT_H
#define METADATAPROPERTIESEDIT_H

#include <QTextEdit>

class MetadataPropertiesEdit : public QTextEdit
{
    Q_OBJECT
public:
    MetadataPropertiesEdit(QWidget *parent = 0);

    void updateMetadata (QString clusterName, QString cellName, quint8 cellColor);
    void requestUpdate ();

signals:
    void metadataPropertiesChanged (QString clusterName, QString cellName, quint8 cellColor);

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
