<?php
    require './helpers.php';

    $db = connectToDB();
    $db->begin_transaction();

    $userName = $_POST["userName"];
    $pw = $_POST["password"];
    $activationCode = $_POST["activationCode"];

    if (!checkPwAndActivationCode($db, $userName, $pw, $activationCode)) {
        echo json_encode(["result"=>false]);
        $db->close();
        exit;
    }

    $success = false;
    if ($db->query("UPDATE user SET ACTIVATION_CODE='' where NAME='" . addslashes($userName) . "'")) {
        $success = true;
        $discordPayload = createNewUserMessage($userName);
        sendDiscordMessage($discordPayload);
    }
    echo json_encode(["result"=>$success]);

    $db->commit();
    $db->close();

    // function for Discord messages
    function createNewUserMessage($userName) {
            return json_encode([
                "username" => "alien-project",
                "avatar_url" => "https://alien-project.org/alien-server/logo.png",
                "content" => "New simulator added to the database",
                "embeds" => [
                    [
                        "author" => [
                            "name" => $userName,
                            "icon_url" => "https://alien-project.org/alien-server/user.png"
                        ],
                    ]
                ]
            ], JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE);
        }
?>