#pragma once

#include <deque>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Network/NetworkResourceRawTO.h>

#include <Persister/Interface/Definitions.h>

#include <Rendering/Interface/TextureData.h>

#include "Definitions.h"

// Preview pictures of the resources shown in the browser. The pictures of the server are requested in batches, the pictures of this
// computer are read from their files, a few per frame.
class _BrowserPictureCache
{
public:
    static BrowserPictureCache create();

    void request(std::vector<NetworkResourceRawTO> const& rawTOs);  // Requests the pictures that are not known yet

    // Null while the picture is being loaded, and without a value if the resource has no picture
    std::optional<TextureData> const* find(NetworkResourceRawTO const& rawTO) const;

    void invalidate(std::string const& resourceId);

    // Forgets the pictures of this computer that are missing, have been written again or belong to none of the given resources, so that
    // changed files are read again
    void retainLocalPictures(std::vector<NetworkResourceRawTO> const& rawTOs);

    void processPendingRequests();

private:
    _BrowserPictureCache();

    struct LocalRequest
    {
        std::string key;
        std::filesystem::path simulationPath;
    };
    struct LocalPicture
    {
        std::optional<TextureData> texture;
        std::filesystem::path picturePath;
        std::filesystem::file_time_type lastChange;  // Of the picture file when it was read
    };

    void requestFromServer(std::vector<std::string> const& resourceIds);

    TaskProcessor _pictureProcessor;
    std::unordered_map<std::string, std::optional<TextureData>> _pictureByResourceId;

    std::unordered_map<std::string, LocalPicture> _localPictureByKey;
    std::deque<LocalRequest> _pendingLocalRequests;
    std::unordered_set<std::string> _pendingLocalKeys;
};
