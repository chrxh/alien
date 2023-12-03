<?php
    require './helpers.php';

    $db = connectToDB();
    $db->begin_transaction();

    $userName = $_POST["userName"];
    $pw = $_POST["newPassword"];
    $salt = base64_encode(mcrypt_create_iv(16, MCRYPT_DEV_URANDOM));
    $pwHash = hash("sha256", $pw . $salt);
    $activationCode = $_POST["activationCode"];

    $obj = $db->query(
        "SELECT 
            u.ACTIVATION_CODE as activationCode
        FROM user u
        WHERE u.NAME='".addslashes($userName)."'")->fetch_object();
    
    if (!$obj) {
        echo json_encode(["result"=>false]);
        $db->close();
        exit;
    }

    if ($obj->activationCode != $activationCode) {
        echo json_encode(["result"=>false]);
        $db->close();
        exit;
    }

    if (!$db->query("UPDATE user SET ACTIVATION_CODE = '', PW_HASH = '".addslashes($pwHash)."', SALT = '".addslashes($salt)."' where NAME='" . addslashes($userName) . "'")) {
        echo json_encode(["result"=>false]);
        $db->close();
        exit;
    }

    echo json_encode(["result"=>true]);

    $db->commit();
    $db->close();
?>