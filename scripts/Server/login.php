<?php
    require './helpers.php';

    $db = connectToDB();
    $db->begin_transaction();

    $userName = $_POST["userName"];
    $pw = $_POST["password"];
    $success = checkPw($db, $userName, $pw);

    if ($success) {
        $gpu = "";
        if (array_key_exists("gpu", $_POST)) {
            $gpu = $_POST["gpu"];
        }
        $success = $db->query("UPDATE user SET FLAGS=1, TIMESTAMP=CURRENT_TIMESTAMP, GPU='".addslashes($gpu)."' WHERE NAME='".addslashes($userName)."'");
    }

    $errorCode = 1;
    
    if ($response = $db->query(
            "SELECT 
                u.ACTIVATION_CODE as activationCode
            FROM user u
            WHERE u.NAME='".addslashes($userName)."'")) {
        if ($obj = $response->fetch_object()) {
            if ($obj->activationCode != "") {
                $errorCode = 0;
            }
        }
    }

    echo json_encode([
        "result" => $success,
        "errorCode" => $errorCode
    ]);
    $db->commit();
    $db->close();
?>
