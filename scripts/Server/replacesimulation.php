<?php
    require './helpers.php';
    require './hooks.php';

    function closeAndExit($db) {
        echo json_encode(["result"=>false]);
        $db->close();
        exit;
    }


    $db = connectToDB();
    $db->begin_transaction();

    $userName = $_POST["userName"];
    $pw = $_POST["password"];

    if (!checkPw($db, $userName, $pw)) {
        closeAndExit($db);
    }

    $obj = $db->query("SELECT u.ID as id FROM user u WHERE u.NAME='".addslashes($userName)."'")->fetch_object();
    if (!$obj) {
        closeAndExit($db);
    }

    $success = false;
    $particles = (int)$_POST['particles'];
    $version = $_POST['version'];
    $width = (int)$_POST['width'];
    $height = (int)$_POST['height'];
    $content = $_POST['content'];
    $settings = $_POST['settings'];
    $simId = $_POST['simId'];
    $size = strlen($content);
    $statistics = array_key_exists('statistics', $_POST) ? $_POST['statistics'] : "";

    $obj = $db->query("SELECT sim.NAME as name, sim.TYPE as type, sim.FROM_RELEASE as workspace FROM simulation sim WHERE sim.ID='".addslashes($simId)."'")->fetch_object();
    if (!$obj) {
        closeAndExit($db);
    }

    if ($userName != 'alien-project' && $obj->workspace == ALIEN_PROJECT_WORKSPACE_TYPE) {
        closeAndExit($db);
    }

    $stmt = $db->prepare("UPDATE simulation SET PARTICLES=?, VERSION=?, CONTENT=?, WIDTH=?, HEIGHT=?, SETTINGS=?, SIZE=?, STATISTICS=?, CONTENT2=?, CONTENT3=?, CONTENT4=?, CONTENT5=?, CONTENT6=? WHERE ID=?");
    if (!$stmt) {
        closeAndExit($db);
    }

    $emptyString = '';
    $stmt->bind_param("issiisissssssi", $particles, $version, $content, $width, $height, $settings, $size, $statistics, $emptyString, $emptyString, $emptyString, $emptyString, $emptyString, $simId);

    if (!$stmt->execute()) {
        closeAndExit($db);
    }

    // create Discord message
    if ($obj->workspace != PRIVATE_WORKSPACE_TYPE) {
        $discordPayload = createUpdateResourceMessage($obj->type, $obj->name, $userName, $simDesc, $width, $height, $particles);
        sendDiscordMessage($discordPayload);
    }

    echo json_encode(["result"=>true]);

    $db->commit();
    $db->close();
?>