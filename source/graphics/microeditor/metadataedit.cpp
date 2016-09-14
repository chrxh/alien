#include "metadataedit.h"
#include "ui_metadataedit.h"

#include "metadatapropertiesedit.h"
#include "../../globaldata/editorsettings.h"

#include <QScrollBar>

MetaDataEdit::MetaDataEdit(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::MetaDataEdit)
{
    ui->setupUi(this);

    QPalette p = ui->cellDescriptionLabel->palette();
    p.setColor(QPalette::WindowText, CELL_EDIT_CAPTION_COLOR1);
    ui->cellDescriptionLabel->setPalette(p);

    p = ui->metaDataDescriptionEdit->palette();
    p.setColor(QPalette::Text, CELL_EDIT_METADATA_CURSOR_COLOR);
    ui->metaDataDescriptionEdit->setPalette(p);

    connect(ui->metaDataPropertiesEdit, SIGNAL(metaDataPropertiesChanged(QString,QString,quint8)), this, SLOT(changesFromMetaDataPropertiesEditor(QString,QString,quint8)));
    connect(ui->metaDataDescriptionEdit, SIGNAL(textChanged()), this, SLOT(changesFromMetaDataDescriptionEditor()));
}

MetaDataEdit::~MetaDataEdit()
{
    delete ui;
}

void MetaDataEdit::updateMetaData (QString clusterName, QString cellName, quint8 cellColor, QString cellDescription)
{
    _clusterName = clusterName;
    _cellName = cellName;
    _cellColor= cellColor;
    _cellDescription = cellDescription;
    ui->metaDataPropertiesEdit->updateMetaData(clusterName, cellName, cellColor);
    ui->metaDataDescriptionEdit->setText(cellDescription);
    if( ui->metaDataDescriptionEdit->verticalScrollBar() )
        ui->metaDataDescriptionEdit->verticalScrollBar()->setValue(0);
}

void MetaDataEdit::requestUpdate ()
{
    _cellDescription = ui->metaDataDescriptionEdit->toPlainText();
    ui->metaDataPropertiesEdit->requestUpdate();
}

void MetaDataEdit::changesFromMetaDataPropertiesEditor (QString clusterName, QString cellName, quint8 cellColor)
{
    _clusterName = clusterName;
    _cellName = cellName;
    _cellColor= cellColor;
    emit metaDataChanged(_clusterName, _cellName, _cellColor, _cellDescription);
}

void MetaDataEdit::changesFromMetaDataDescriptionEditor ()
{
    _cellDescription = ui->metaDataDescriptionEdit->toPlainText();
    emit metaDataChanged(_clusterName, _cellName, _cellColor, _cellDescription);
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
