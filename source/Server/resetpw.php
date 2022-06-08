<?php
    require './helpers.php';

    $db = connectToDB();
    $db->begin_transaction();

    $userName = $_POST["userName"];
    $email = $_POST["email"];
    $emailHash = hash("sha256", $email);

    $obj = $db->query(
        "SELECT 
            u.EMAIL_HASH as emailHash
        FROM user u
        WHERE u.NAME='".addslashes($userName)."'")->fetch_object();
    
    if (!$obj) {
        echo json_encode(["result"=>false]);
        $db->close();
        exit;
    }

    if ($obj->emailHash != $emailHash) {
        echo json_encode(["result"=>false]);
        $db->close();
        exit;
    }

    $activationCode = bin2hex(mcrypt_create_iv(3, MCRYPT_DEV_URANDOM));
    if (!$db->query("UPDATE user SET ACTIVATION_CODE='$activationCode' where NAME='" . addslashes($userName) . "'")) {
        echo json_encode(["result"=>false]);
        $db->close();
        exit;
    }

    mail(
        $email,
        "Artificial Life Environment: confirmation code for user '" . addslashes($userName) . "'",
        "Your confirmation code for user '".addslashes($userName)."' is:\n\n" . $activationCode,
        "From: user registration <info@alien-project.org>");

    echo json_encode(["result"=>true]);

    $db->commit();
    $db->close();
?>