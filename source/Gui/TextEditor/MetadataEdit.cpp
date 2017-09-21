#include <QScrollBar>

#include "Gui/Settings.h"
#include "MetadataPropertiesEdit.h"

#include "MetadataEdit.h"
#include "ui_MetadataEdit.h"

MetadataEdit::MetadataEdit(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::MetadataEdit)
{
    ui->setupUi(this);

    QPalette p = ui->cellDescriptionLabel->palette();
    p.setColor(QPalette::WindowText, CELL_EDIT_CAPTION_COLOR1);
    ui->cellDescriptionLabel->setPalette(p);

    p = ui->metadataDescriptionEdit->palette();
    p.setColor(QPalette::Text, CELL_EDIT_METADATA_CURSOR_COLOR);
    ui->metadataDescriptionEdit->setPalette(p);

    connect(ui->metadataPropertiesEdit, SIGNAL(metadataPropertiesChanged(QString,QString,quint8)), this, SLOT(changesFromMetadataPropertiesEditor(QString,QString,quint8)));
    connect(ui->metadataDescriptionEdit, SIGNAL(textChanged()), this, SLOT(changesFromMetadataDescriptionEditor()));
}

MetadataEdit::~MetadataEdit()
{
    delete ui;
}

void MetadataEdit::updateMetadata (QString clusterName, QString cellName, quint8 cellColor, QString cellDescription)
{
    _clusterName = clusterName;
    _cellName = cellName;
    _cellColor= cellColor;
    _cellDescription = cellDescription;
    ui->metadataPropertiesEdit->updateMetadata(clusterName, cellName, cellColor);
    ui->metadataDescriptionEdit->setText(cellDescription);
    if( ui->metadataDescriptionEdit->verticalScrollBar() )
        ui->metadataDescriptionEdit->verticalScrollBar()->setValue(0);
}

void MetadataEdit::requestUpdate ()
{
    _cellDescription = ui->metadataDescriptionEdit->toPlainText();
    ui->metadataPropertiesEdit->requestUpdate();
}

void MetadataEdit::changesFromMetadataPropertiesEditor (QString clusterName, QString cellName, quint8 cellColor)
{
    _clusterName = clusterName;
    _cellName = cellName;
    _cellColor= cellColor;
    Q_EMIT metadataChanged(_clusterName, _cellName, _cellColor, _cellDescription);
}

void MetadataEdit::changesFromMetadataDescriptionEditor ()
{
    _cellDescription = ui->metadataDescriptionEdit->toPlainText();
    Q_EMIT metadataChanged(_clusterName, _cellName, _cellColor, _cellDescription);
}

/*
<property name="frameShape">
 <enum>QFrame::NoFrame</enum>
</property>
<property name="frameShadow">
 <enum>QFrame::Plain</enum>
</property>
<property name="lineWidth">
 <number>0</number>
</property>
<property name="cursorWidth">
 <number>6</number>
</property>
*/
