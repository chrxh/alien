<?php
    require './helpers.php';

    $db = connectToDB();
    $db->begin_transaction();

    $userName = $_POST["userName"];
    $pw = $_POST["password"];

    if (!checkPw($db, $userName, $pw)) {
        echo json_encode(["result"=>false]);
        $db->close();
        exit;
    }

    $obj = $db->query("SELECT u.ID as id FROM user u WHERE u.NAME='".addslashes($userName)."'")->fetch_object();
    if (!$obj) {
        echo json_encode(["result"=>false]);
        $db->close();
        exit;
    }

    $success = false;
    $simName = $_POST['simName'];
    $simDesc = $_POST['simDesc'];
    $width = (int)$_POST['width'];
    $height = (int)$_POST['height'];
    $particles = (int)$_POST['particles'];
    $version = $_POST['version'];
    $content = $_POST['content'];
    $settings = $_POST['settings'];
    $symbolMap = $_POST['symbolMap'];
    $size = strlen($content);
    $type = array_key_exists("type", $_POST) ? $_POST['type'] : 0;

    if ($db->query("INSERT INTO simulation (ID, USER_ID, NAME, WIDTH, HEIGHT, PARTICLES, VERSION, DESCRIPTION, CONTENT, SETTINGS, SYMBOL_MAP, PICTURE, TIMESTAMP, SIZE, TYPE)
                    VALUES (NULL, {$obj->id}, '" . addslashes($simName) . "', $width, $height, $particles, '" . addslashes($version) . "', '" . addslashes($simDesc) . "', '" . addslashes($content) . "', '" . addslashes($settings) . "', '" . addslashes($symbolMap) . "', 'a', NULL, $size, $type)")) {
        $success = true;

        // create Discord message
        if ($type == 0) {
            $discordPayload = createAddSimulationMessage($simName, $userName, $simDesc, $width, $height, $particles);
        }
        if ($type == 1) {
            $discordPayload = createAddGenomeMessage($simName, $userName, $simDesc, $width, $height, $particles);
        }
        sendDiscordMessage($discordPayload);
    }

    echo json_encode(["result"=>$success]);

    $db->commit();
    $db->close();

    // functions for Discord messages
    function createAddSimulationMessage($simName, $userName, $simDesc, $width, $height, $particles) {
        return json_encode([
            "username" => "alien-project",
            "avatar_url" => "https://alien-project.org/alien-server/logo.png",
            "content" => "New simulation added to the database",
            "embeds" => [
                [
                    "author" => [
                        "name" => $simName,
                        "icon_url" => "https://alien-project.org/alien-server/galaxy.png"
                    ],
                    "title" => "By " . $userName,
                    "description" => $simDesc,
                    "fields" => [
                        [
                          "name" => "Size",
                          "value" => "{$width} x {$height}",
                          "inline" => true
                        ],
                        [
                          "name" => "Objects",
                          "value" => strval((int)($particles/1000)) . " K",
                          "inline" => true
                        ]
                    ]
                ]
            ]
        ], JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE);
    }

    function createAddGenomeMessage($simName, $userName, $simDesc, $width, $height, $particles) {
        return json_encode([
            "username" => "alien-project",
            "avatar_url" => "https://alien-project.org/alien-server/logo.png",
            "content" => "New genome added to the database",
            "embeds" => [
                [
                    "author" => [
                        "name" => $simName,
                        "icon_url" => "https://alien-project.org/alien-server/genome.png"
                    ],
                    "title" => "By " . $userName,
                    "description" => $simDesc,
                    "fields" => [
                        [
                          "name" => "Cells",
                          "value" => "{$particles}",
                          "inline" => true
                        ]
                    ]
                ]
            ]
        ], JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE);
    }
?>