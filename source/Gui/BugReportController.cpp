#include "BugReportController.h"

#include <QEventLoop>
#include <QMessageBox>
#include <QProgressDialog>

#include "Base/ServiceLocator.h"
#include "Web/WebAccess.h"
#include "Web/WebBuilderFacade.h"

#include "MessageHelper.h"
#include "BugReportView.h"

BugReportController::BugReportController(std::string const& errorMessage, std::string const& protocol)
{
    _view = new BugReportView(errorMessage, protocol);

    auto facade = ServiceLocator::getInstance().getService<WebBuilderFacade>();
    auto webAccess = facade->buildWebAccess();
    SET_CHILD(_webAccess, webAccess);
}

BugReportController::~BugReportController()
{
    delete _view;
}

void BugReportController::execute()
{
    if (_view->exec()) {
        auto protocol = _view->getProtocol();
        auto email = _view->getEmailAddress();
        auto message = _view->getUserMessage();

        auto progress = MessageHelper::createProgressDialog("Sending report...", _view);

        QEventLoop pause;
        bool finished = false;
        _webAccess->connect(_webAccess, &WebAccess::error, [&]() {
            finished = true;
            delete progress;

            QMessageBox messageBox;
            messageBox.critical(0, "Critical error", "The server does not respond. Your bug report could not be sent.");
            pause.quit();
        });
        _webAccess->connect(_webAccess, &WebAccess::sendBugReportReceived, [&]() {
            finished = true;
            delete progress;

            QMessageBox messageBox;
            messageBox.information(0, "Success", "Your bug report was sent successfully.");
            pause.quit();
        });

        _webAccess->sendBugReport(protocol, email, message);
        if (!finished) {
            pause.exec();
        }
    }
}
