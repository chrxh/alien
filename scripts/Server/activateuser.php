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
        if (array_key_exists("gpu", $_POST)) {
            $gpu = $_POST["gpu"];
            $discordPayload = createNewUserMessageWithGpuInfo($userName, $gpu);
        }
        else {
            $discordPayload = createNewUserMessageWithoutGpuInfo($userName);
        }
        sendDiscordMessage($discordPayload);
    }
    echo json_encode(["result"=>$success]);

    $db->commit();
    $db->close();

    // function for Discord messages
    function createNewUserMessageWithGpuInfo($userName, $gpu) {
        return json_encode([
            "username" => "alien-project",
            "avatar_url" => "https://alien-project.org/alien-server/logo.png",
            "content" => "",
            "embeds" => [
                [
                    "author" => [
                        "name" => "New simulator added to the database",
                        "icon_url" => "https://alien-project.org/alien-server/userpic.png"
                    ],
                    "fields" => [
                        [
                            "name" => "Name",
                            "value" => $userName,
                            "inline" => true
                        ],
                        [
                          "name" => "Powered by",
                          "value" => $gpu,
                          "inline" => true
                        ]
                    ]
                ]
            ]
        ], JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE);
    }

    function createNewUserMessageWithoutGpuInfo($userName) {
        return json_encode([
            "username" => "alien-project",
            "avatar_url" => "https://alien-project.org/alien-server/logo.png",
            "content" => "",
            "embeds" => [
                [
                    "author" => [
                        "name" => "New simulator added to the database",
                        "icon_url" => "https://alien-project.org/alien-server/userpic.png"
                    ],
                    "fields" => [
                        [
                            "name" => "Name",
                            "value" => $userName,
                            "inline" => true
                        ]
                    ]
                ]
            ]
        ], JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE);
    }
?>