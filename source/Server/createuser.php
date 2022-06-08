<?php
    require './helpers.php';

    $db = connectToDB();
    $db->begin_transaction();

    $userName = $_POST["userName"];
    if (!preg_match("#^[^ ]+$#", $userName)) {
        echo json_encode(["result"=>false]);
        exit;
    }

    $pw = $_POST["password"];
    $email = str_replace(" ", "", $_POST["email"]);

    $salt = base64_encode(mcrypt_create_iv(16, MCRYPT_DEV_URANDOM));
    $pwHash = hash("sha256", $pw . $salt);
    $emailHash = hash("sha256", $email);
    $activationCode = bin2hex(mcrypt_create_iv(3, MCRYPT_DEV_URANDOM));

    $obj = $db->query(
        "SELECT 
            u.PW_HASH as pwHash,
            u.SALT as salt,
            u.ACTIVATION_CODE as activationCode
        FROM user u
        WHERE u.NAME='".addslashes($userName)."'")->fetch_object();
    if ($obj && $obj->activationCode != "") {
        if (!$db->query("DELETE FROM user WHERE NAME='".addslashes($userName)."'")) {
            echo json_encode(["result"=>false]);
            $db->close();
            exit;
        }
    }

    $success = false;
    if ($db->query("INSERT INTO user (ID, NAME, PW_HASH, EMAIL_HASH, SALT, ACTIVATION_CODE, FLAGS, TIMESTAMP) 
        VALUES (NULL, '".addslashes($userName)."', '".addslashes($pwHash)."', '".addslashes($emailHash)."', '".addslashes($salt)."', '$activationCode', 0, NULL)")) {
        $success = true;

        mail(
            $email,
            "Artificial Life Environment: confirmation code for user '" . addslashes($userName) . "'",
            "Your confirmation code for user '".addslashes($userName)."' is:\n\n" . $activationCode,
            "From: user registration <info@alien-project.org>");
    }
    echo json_encode(["result"=>$success]);

    $db->commit();
    $db->close();
?>