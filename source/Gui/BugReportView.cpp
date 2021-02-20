#include "BugReportView.h"
#include "ui_BugReportDialog.h"

#include "Settings.h"

BugReportView::BugReportView(std::string const& errorMessage, std::string const& protocol, QWidget* parent)
    : QDialog(parent)
    , ui(new Ui::BugReportDialog)
{
    ui->setupUi(this);
    ui->errorMessageLabel->setText(QString::fromStdString(errorMessage));
    ui->protocolTextEdit->setText(QString::fromStdString(protocol));
}

BugReportView::~BugReportView()
{
    delete ui;
}

std::string BugReportView::getProtocol() const
{
    return ui->protocolTextEdit->toPlainText().toStdString();
}

std::string BugReportView::getEmailAddress() const
{
    return ui->emailAddressEdit->text().toStdString();
}

std::string BugReportView::getUserMessage() const
{
    return ui->userMessageEdit->toPlainText().toStdString();
}
