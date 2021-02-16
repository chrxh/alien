#include "LoggingView.h"

#include "Settings.h"
#include "ui_LoggingView.h"

LoggingView::LoggingView(QWidget* parent /*= nullptr*/)
    : QWidget(parent)
    , ui(new Ui::LoggingView)
{
    ui->setupUi(this);
}

LoggingView::~LoggingView()
{
    delete ui;
}

void LoggingView::setNewLogMessage(std::string const& message)
{
    QString colorTextStart = "<span style=\"color:" + Const::CellEditTextColor1.name() + "\">";
    QString colorEnd = "</span>";

    auto text = ui->contentLabel->text();
    text = colorTextStart + QString::fromStdString(message) + colorEnd + "<br/>" + text;
    ui->contentLabel->setText(text);
    repaint();
}
