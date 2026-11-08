#include "BrowserPictureCache.h"

#include <Base/LoggingService.h>
#include <Base/StringHelper.h>

#include <Persister/Interface/PersisterFacade.h>
#include <Persister/Interface/SimulationLibraryService.h>
#include <Persister/Interface/TaskProcessor.h>

#include <Rendering/Interface/RenderingFacade.h>

#include "BrowserData.h"

namespace
{
    auto constexpr MaxPicturesPerRequest = 32;
    auto constexpr MaxLocalPicturesPerFrame = 2;

    // A simulation that is saved again gets a new picture under the same path
    std::string getLocalKey(NetworkResourceRawTO const& rawTO)
    {
        return rawTO->id + "|" + rawTO->timestamp;
    }

    std::optional<TextureData> decodePicture(std::string const& jpg, std::string const& resourceId)
    {
        try {
            return _RenderingFacade::get()->loadTextureFromMemory(jpg);
        } catch (std::exception const&) {
            log(Priority::Important, "browser: preview picture of simulation " + resourceId + " could not be decoded");
            return std::nullopt;
        }
    }
}

BrowserPictureCache _BrowserPictureCache::create()
{
    return BrowserPictureCache(new _BrowserPictureCache());
}

_BrowserPictureCache::_BrowserPictureCache()
{
    _pictureProcessor = _TaskProcessor::createTaskProcessor(_PersisterFacade::get());
}

void _BrowserPictureCache::request(std::vector<NetworkResourceRawTO> const& rawTOs)
{
    std::vector<std::string> resourceIds;
    for (auto const& rawTO : rawTOs) {
        if (isLocalWorkspace(rawTO->workspaceType)) {
            auto key = getLocalKey(rawTO);
            if (!_localPictureByKey.contains(key) && _pendingLocalKeys.insert(key).second) {
                _pendingLocalRequests.emplace_back(LocalRequest{.key = key, .simulationPath = StringHelper::toPath(rawTO->id)});
            }
        } else if (!_pictureByResourceId.contains(rawTO->id) && toInt(resourceIds.size()) < MaxPicturesPerRequest) {
            resourceIds.emplace_back(rawTO->id);
        }
    }
    if (!resourceIds.empty() && !_pictureProcessor->pendingTasks()) {
        requestFromServer(resourceIds);
    }
}

std::optional<TextureData> const* _BrowserPictureCache::find(NetworkResourceRawTO const& rawTO) const
{
    if (isLocalWorkspace(rawTO->workspaceType)) {
        auto findResult = _localPictureByKey.find(getLocalKey(rawTO));
        return findResult != _localPictureByKey.end() ? &findResult->second.texture : nullptr;
    }
    auto findResult = _pictureByResourceId.find(rawTO->id);
    return findResult != _pictureByResourceId.end() ? &findResult->second : nullptr;
}

void _BrowserPictureCache::invalidate(std::string const& resourceId)
{
    auto findResult = _pictureByResourceId.find(resourceId);
    if (findResult == _pictureByResourceId.end()) {
        return;
    }
    if (findResult->second.has_value()) {
        _RenderingFacade::get()->deleteTexture(*findResult->second);
    }
    _pictureByResourceId.erase(findResult);
}

void _BrowserPictureCache::retainLocalPictures(std::vector<NetworkResourceRawTO> const& rawTOs)
{
    std::unordered_set<std::string> keys;
    for (auto const& rawTO : rawTOs) {
        keys.insert(getLocalKey(rawTO));
    }
    std::erase_if(_pendingLocalRequests, [&](LocalRequest const& request) { return !keys.contains(request.key); });
    std::erase_if(_pendingLocalKeys, [&](std::string const& key) { return !keys.contains(key); });
    std::erase_if(_localPictureByKey, [&](auto const& entry) {
        auto const& [key, picture] = entry;
        if (!picture.texture.has_value()) {
            return true;
        }

        // The time of a simulation file has a resolution of seconds in the key, so the picture file tells whether it has been written again
        std::error_code error;
        auto isCurrent = keys.contains(key) && std::filesystem::last_write_time(picture.picturePath, error) == picture.lastChange && !error;
        if (isCurrent) {
            return false;
        }
        _RenderingFacade::get()->deleteTexture(*picture.texture);
        return true;
    });
}

void _BrowserPictureCache::processPendingRequests()
{
    _pictureProcessor->process();

    auto numLoadedPictures = 0;
    while (!_pendingLocalRequests.empty() && numLoadedPictures++ < MaxLocalPicturesPerFrame) {
        auto request = _pendingLocalRequests.front();
        _pendingLocalRequests.pop_front();
        _pendingLocalKeys.erase(request.key);

        // The time is read before the picture, so that a picture that is replaced meanwhile is read again later
        auto& service = SimulationLibraryService::get();
        auto picturePath = service.getPicturePath(request.simulationPath);
        std::error_code error;
        auto lastChange = std::filesystem::last_write_time(picturePath, error);

        auto jpg = service.loadPicture(request.simulationPath);
        auto hasPicture = jpg.has_value() && !jpg->empty();
        _localPictureByKey.insert_or_assign(
            request.key,
            LocalPicture{.texture = hasPicture ? decodePicture(*jpg, request.key) : std::nullopt, .picturePath = picturePath, .lastChange = lastChange});
    }
}

void _BrowserPictureCache::requestFromServer(std::vector<std::string> const& resourceIds)
{
    _pictureProcessor->executeTask(
        [resourceIds](auto const& senderId) {
            return _PersisterFacade::get()->scheduleGetSimulationPictures(
                SenderInfo{.senderId = senderId, .wishResultData = true, .wishErrorInfo = true}, GetSimulationPicturesRequestData{.simIds = resourceIds});
        },
        [&, resourceIds](auto const& requestId) {
            auto data = _PersisterFacade::get()->fetchGetSimulationPicturesData(requestId);
            for (auto const& resourceId : resourceIds) {
                auto findResult = data.jpgBySimId.find(resourceId);
                auto hasPicture = findResult != data.jpgBySimId.end() && !findResult->second.empty();
                _pictureByResourceId.insert_or_assign(resourceId, hasPicture ? decodePicture(findResult->second, resourceId) : std::nullopt);
            }
        },
        [&, resourceIds](auto const&) {
            for (auto const& resourceId : resourceIds) {
                _pictureByResourceId.insert_or_assign(resourceId, std::nullopt);
            }
        });
}
