#pragma once

#include "Base/Singleton.h"
#include "EngineInterface/Definitions.h"
#include "PersisterInterface/PersisterErrorInfo.h"

#include "Definitions.h"
#include "MainLoopEntity.h"
#include "AlienDialog.h"

class MessageDialog : public AlienDialog<>
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(MessageDialog);

public:
    void information(std::string const& title, std::string const& message);
    void information(std::string const& title, std::vector<PersisterErrorInfo> const& errors);
    void yesNo(std::string const& title, std::string const& message, std::function<void()> const& yesFunction);

private:
    MessageDialog();

    void open() override {}
    void processIntern() override;

    void processInformation();
    void processYesNo();

    bool _sizeInitialized = false;

    enum class DialogType
    {
        Information, YesNo
    };
    DialogType _dialogType = DialogType::Information;
    std::string _title;
    std::string _message;
    std::function<void()> _execFunction;
};

inline void showMessage(std::string const& title, std::string const& message)
{
    MessageDialog::get().information(title, message);
}
